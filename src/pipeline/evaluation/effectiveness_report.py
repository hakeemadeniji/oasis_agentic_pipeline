"""
Effectiveness & Data Analysis report generator for the OASIS Agentic Pipeline.

Runs a real, reproducible evaluation of the pipeline and renders the results to
``docs/OASIS_Pipeline_Effectiveness_Analysis.pdf``:

  * Vision agent classification metrics on the OASIS MRI dataset
    (confusion matrix, per-class precision/recall/F1, balanced accuracy, kappa,
    and a clinically-meaningful "impaired vs non-demented" screening collapse).
  * Cohort data analysis from the OASIS clinical + longitudinal CSVs
    (temporal-agent atrophy-velocity separation across diagnostic groups).
  * Ethicist guardrail effectiveness on a labelled adversarial battery.
  * Regional volumetry demonstration (FreeSurfer aseg z-scoring).
  * Edge inference benchmark across ONNX Runtime providers
    (Snapdragon NPU / DirectML / CPU) and INT8 size reduction.

Usage:
    python src/pipeline/evaluation/effectiveness_report.py [--max-per-class 60]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import Dict, List, Tuple

import numpy as np

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, "..", ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
if _SRC not in sys.path:
    sys.path.append(_SRC)

from pipeline.evaluation.pil_report import (  # noqa: E402
    ReportBuilder,
    GOOD,
    WARN,
    BAD,
    ACCENT,
    ACCENT2,
)

CLASS_NAMES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]


# ===========================================================================
# 1. Vision classification evaluation
# ===========================================================================
def evaluate_vision(root: str, max_per_class: int, weights_path: str | None = None) -> Dict:
    import torch
    from torchvision import transforms
    from PIL import Image
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
    )
    from agents.vision.vision_agent import AlzheimerVisionAgent

    data_root = os.path.join(root, "data", "oasis_raw")
    weights = weights_path or os.path.join(
        root, "src", "pipeline", "onnx_inference", "best_vision_agent.pth"
    )

    # Match the training-time val/test transform exactly (incl. normalization).
    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # PATIENT-GROUPED evaluation: score ONLY on held-out test SUBJECTS the model
    # never saw in training (no subject overlap). This reflects real generalization
    # rather than memorized per-patient slices (image-level splits inflate accuracy
    # because there are ~240 near-identical slices per subject).
    from pipeline.data_split import patient_grouped_split, subject_id_from_path

    classes, _train, _val, test_items = patient_grouped_split(data_root, seed=42)
    by_class: Dict[int, List[Tuple[str, int]]] = {}
    for path, cls in test_items:
        by_class.setdefault(cls, []).append((path, cls))
    rng = np.random.default_rng(123)
    sampled: List[Tuple[str, int]] = []
    for cls in sorted(by_class):
        lst = by_class[cls]
        idx = rng.permutation(len(lst))[:max_per_class]
        sampled += [lst[i] for i in idx]
    # Per-class count of held-out test SUBJECTS (transparency: a class with very
    # few subjects yields a statistically meaningless per-class metric).
    _subj: Dict[int, set] = {}
    for path, cls in test_items:
        _subj.setdefault(cls, set()).add(subject_id_from_path(path))
    test_subjects = {c: len(s) for c, s in _subj.items()}

    device = torch.device("cpu")
    model = AlzheimerVisionAgent(num_classes=len(classes)).to(device)
    trained = os.path.exists(weights)
    if trained:
        model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    y_true, y_pred, confidences = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        batch, labels = [], []
        for path, cls in sampled:
            img = Image.open(path).convert("L")
            batch.append(tf(img))
            labels.append(cls)
            if len(batch) == 32:
                x = torch.stack(batch).to(device)
                probs = torch.softmax(model(x), dim=1)
                conf, pred = probs.max(dim=1)
                y_pred.extend(pred.tolist())
                y_true.extend(labels)
                confidences.extend(conf.tolist())
                batch, labels = [], []
        if batch:
            x = torch.stack(batch).to(device)
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
            y_pred.extend(pred.tolist())
            y_true.extend(labels)
            confidences.extend(conf.tolist())
    elapsed = time.perf_counter() - t0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Binary screening collapse: impaired (any dementia) vs non-demented.
    non_idx = classes.index("Non Demented") if "Non Demented" in classes else 2
    yb_true = [0 if t == non_idx else 1 for t in y_true]
    yb_pred = [0 if p == non_idx else 1 for p in y_pred]
    tp = sum(1 for t, p in zip(yb_true, yb_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(yb_true, yb_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(yb_true, yb_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yb_true, yb_pred) if t == 0 and p == 1)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "trained": trained,
        "classes": classes,
        "split": "patient_grouped",
        "test_subjects": test_subjects,
        "n": len(y_true),
        "elapsed_s": elapsed,
        "throughput": len(y_true) / elapsed if elapsed else 0.0,
        "cm": cm.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "sensitivity": sens,
        "specificity": spec,
    }


# ===========================================================================
# 2. Cohort + longitudinal data analysis
# ===========================================================================
def analyze_cohort(root: str) -> Dict:
    import pandas as pd
    from agents.biomarker.temporal_analyst import TemporalAnalystAgent

    out: Dict = {}
    clinical = os.path.join(root, "data", "oasis_raw", "oasis_clinical_data.csv")
    longitudinal = os.path.join(root, "data", "oasis_raw", "oasis_longitudinal.csv")

    if os.path.exists(clinical):
        df = pd.read_csv(clinical)
        out["clinical_n"] = len(df)
        out["age_mean"] = float(df["Age"].mean())
        out["age_std"] = float(df["Age"].std())
        out["mmse_mean"] = float(df["MMSE"].mean())
        out["mmse_std"] = float(df["MMSE"].std())
        out["nwbv_mean"] = float(df["nWBV"].mean())
        if "M/F" in df.columns:
            vc = df["M/F"].value_counts()
            out["sex"] = {str(k): int(v) for k, v in vc.items()}

    if os.path.exists(longitudinal):
        ldf = pd.read_csv(longitudinal)
        out["long_n"] = len(ldf)
        analyst = TemporalAnalystAgent(longitudinal)
        # Atrophy velocity by diagnostic group (temporal-agent effectiveness).
        group_vel: Dict[str, List[float]] = {}
        for sid in ldf["Subject ID"].unique():
            sub = ldf[ldf["Subject ID"] == sid]
            grp = str(sub["Group"].iloc[0]) if "Group" in sub.columns else "Unknown"
            try:
                metrics = analyst.calculate_progression_trajectory(sid)
            except Exception:
                continue
            if metrics.get("visits_tracked", 0) > 1:
                group_vel.setdefault(grp, []).append(
                    float(metrics.get("atrophy_velocity_pct", 0.0))
                )
        out["atrophy_by_group"] = {
            g: {"mean": float(np.mean(v)), "n": len(v)} for g, v in group_vel.items() if v
        }
    return out


# ===========================================================================
# 3. Ethicist guardrail effectiveness (labelled battery)
# ===========================================================================
def evaluate_guardrail(root: str) -> Dict:
    from orchestrator.ethicist_agent import MedicalEthicistAgent

    ethicist = MedicalEthicistAgent(confidence_floor=60.0)
    # (name, predicted_class, confidence, mmse, atrophy_velocity, expected_flagged)
    battery = [
        ("Consistent: healthy", "Non Demented", 92.0, 29.0, 0.3, False),
        ("Consistent: moderate AD", "Moderate Dementia", 88.0, 15.0, 1.3, False),
        ("Consistent: mild AD", "Mild Dementia", 80.0, 22.0, 0.9, False),
        ("Low model confidence", "Non Demented", 51.0, 28.0, 0.4, True),
        ("Cross-modal contradiction", "Moderate Dementia", 90.0, 29.0, 0.4, True),
        ("Dangerous type-II (missed AD)", "Non Demented", 90.0, 8.0, 0.6, True),
        ("Silent degradation (warn)", "Non Demented", 90.0, 29.0, 2.6, False),
        ("Mild claim, perfect cognition", "Mild Dementia", 85.0, 28.0, 0.5, True),
    ]
    rows = []
    correct = 0
    for name, cls, conf, mmse, vel, expected in battery:
        flagged, msg = ethicist.audit_diagnostic_proposal(cls, conf, mmse, vel)
        ok = flagged == expected
        correct += int(ok)
        rows.append(
            (
                name,
                "FLAG" if flagged else "pass",
                "FLAG" if expected else "pass",
                "ok" if ok else "MISS",
            )
        )
    return {"rows": rows, "accuracy": correct / len(battery), "n": len(battery)}


# ===========================================================================
# 4. Regional volumetry demonstration
# ===========================================================================
def evaluate_volumetry(root: str) -> Dict:
    import tempfile
    from agents.biomarker.volumetry_agent import RegionalVolumetryAgent

    # Healthy vs atrophic synthetic aseg to show z-score separation.
    def aseg(hippo, vent):
        return f"""# Measure EstimatedTotalIntraCranialVol, eTIV, x, 1500000.0, mm^3
# ColHeaders Index SegId NVoxels Volume_mm3 StructName
1 4 1 {vent} Left-Lateral-Ventricle
2 43 1 {vent} Right-Lateral-Ventricle
3 17 1 {hippo} Left-Hippocampus
4 53 1 {hippo} Right-Hippocampus
5 18 1 {hippo * 0.45:.0f} Left-Amygdala
6 54 1 {hippo * 0.45:.0f} Right-Amygdala
"""

    agent = RegionalVolumetryAgent()
    results = {}
    for label, hippo, vent in [
        ("Healthy reference", 3550, 15000),
        ("Atrophic (AD-like)", 2350, 33000),
    ]:
        with tempfile.NamedTemporaryFile("w", suffix="_aseg.stats", delete=False) as fh:
            fh.write(aseg(hippo, vent))
            tmp = fh.name
        r = agent.analyze_stats_file(tmp, subject_id=label)
        results[label] = {
            "mta_risk": r.mta_risk_score,
            "stage": r.mta_stage,
            "lh_z": next((m.z_score for m in r.regions if m.structure == "Left-Hippocampus"), 0.0),
            "vent_z": next(
                (m.z_score for m in r.regions if m.structure == "Left-Lateral-Ventricle"), 0.0
            ),
        }
        os.remove(tmp)
    return results


# ===========================================================================
# 5. Edge inference benchmark across ONNX providers
# ===========================================================================
def benchmark_inference(root: str) -> Dict:
    out: Dict = {"providers": [], "torch_ms": None, "sizes": {}}
    onnx_dir = os.path.join(root, "src", "pipeline", "onnx_inference")
    raw = os.path.join(onnx_dir, "multimodal_fusion.onnx")
    int8 = os.path.join(onnx_dir, "multimodal_fusion_int8.onnx")
    raw_data = os.path.join(onnx_dir, "multimodal_fusion.onnx.data")

    def mb(p):
        return os.path.getsize(p) / (1024 * 1024) if os.path.exists(p) else 0.0

    raw_sz = mb(raw) + mb(raw_data)
    int8_sz = mb(int8)
    out["sizes"] = {
        "raw_mb": raw_sz,
        "int8_mb": int8_sz,
        "reduction_pct": (1 - int8_sz / raw_sz) * 100 if raw_sz else 0.0,
    }

    # torch baseline latency
    try:
        import torch
        from agents.vision.fusion_agent import AlzheimerMultimodalFusionNet

        m = AlzheimerMultimodalFusionNet().eval()
        img = torch.randn(1, 1, 224, 224)
        tab = torch.randn(1, 8)
        with torch.no_grad():
            for _ in range(3):
                m(img, tab)
            t0 = time.perf_counter()
            for _ in range(20):
                m(img, tab)
            out["torch_ms"] = (time.perf_counter() - t0) / 20 * 1000
    except Exception as e:
        out["torch_error"] = str(e)

    # ONNX Runtime per-provider latency on the INT8 edge artifact
    model_path = int8 if os.path.exists(int8) else (raw if os.path.exists(raw) else None)
    if model_path:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            wanted = [
                "QNNExecutionProvider",
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            label = {
                "QNNExecutionProvider": "Snapdragon NPU (QNN)",
                "DmlExecutionProvider": "ARM64 GPU (DirectML)",
                "CUDAExecutionProvider": "GPU (CUDA)",
                "CPUExecutionProvider": "CPU",
            }
            for prov in wanted:
                if prov not in available:
                    continue
                try:
                    sess = ort.InferenceSession(model_path, providers=[prov])
                    names = [i.name for i in sess.get_inputs()]
                    feeds = {names[0]: np.random.randn(1, 1, 224, 224).astype(np.float32)}
                    if len(names) > 1:
                        feeds[names[1]] = np.random.randn(1, 8).astype(np.float32)
                    for _ in range(3):
                        sess.run(None, feeds)
                    t0 = time.perf_counter()
                    for _ in range(20):
                        sess.run(None, feeds)
                    ms = (time.perf_counter() - t0) / 20 * 1000
                    out["providers"].append(
                        {"provider": label.get(prov, prov), "ms": ms, "active": True}
                    )
                except Exception as e:
                    out["providers"].append(
                        {"provider": label.get(prov, prov), "ms": None, "error": str(e)[:60]}
                    )
        except Exception as e:
            out["onnx_error"] = str(e)
    return out


# ===========================================================================
# Report assembly
# ===========================================================================
def build_report(
    root: str,
    vision: Dict,
    cohort: Dict,
    guardrail: Dict,
    volumetry: Dict,
    bench: Dict,
    out_path: str,
) -> str:
    rb = ReportBuilder(
        title="OASIS Agentic Pipeline — Effectiveness Analysis",
        subtitle=f"Automated evaluation report generated {date.today().isoformat()}.",
    )
    rb.cover(
        [
            "RESEARCH USE ONLY - NOT a medical device; not for clinical or diagnostic use.",
            f"MRI scans evaluated: {vision.get('n', 0)} from subject-disjoint HELD-OUT patients "
            f"(no subject overlap with training) across {len(vision.get('classes', []))} classes",
            f"Clinical cohort: {cohort.get('clinical_n', 0)} subjects | Longitudinal records: {cohort.get('long_n', 0)}",
            "Acceleration: ONNX Runtime QNN (Snapdragon NPU) → DirectML → CPU fallback",
            "Language reasoning: local Ollama / cost-tiered Claude (hybrid)",
        ]
    )

    # --- Executive summary ---
    rb.new_page()
    rb.heading("1. Executive Summary", 1)
    rb.kpi_row(
        [
            ("Overall accuracy", f"{vision.get('accuracy', 0) * 100:.1f}%", ACCENT),
            ("Balanced acc.", f"{vision.get('balanced_accuracy', 0) * 100:.1f}%", ACCENT2),
            ("Screening sensitivity", f"{vision.get('sensitivity', 0) * 100:.1f}%", GOOD),
            ("Guardrail acc.", f"{guardrail.get('accuracy', 0) * 100:.1f}%", WARN),
        ]
    )
    rb.paragraph(
        "RESEARCH USE ONLY. This is a research prototype and screening decision-support "
        "demonstrator - NOT a medical device, and not cleared by any regulator (FDA/CE). "
        "It must not be used for clinical diagnosis or patient management.",
        color=BAD,
    )
    rb.paragraph(
        "Validation note: metrics below are computed on a SUBJECT-DISJOINT held-out test set "
        "(no patient appears in both training and test) - the honest measure of generalization. "
        "The bundled OASIS-1 slice set is patient-poor in some classes (e.g. Moderate Dementia "
        "has only ~2 subjects total), so small-class metrics are noisy and indicative only; "
        "clinical-grade claims require a large multi-site cohort (OASIS-3 / ADNI).",
        color=WARN,
    )
    rb.paragraph(
        "This report quantifies the effectiveness of the OASIS Agentic Pipeline, a hybrid "
        "edge-cloud multi-agent system for Alzheimer's disease screening. The vision agent "
        "(ResNet18) classifies MRI slices into four severity classes; a clinical-biomarker "
        "agent, a longitudinal temporal analyst, a regional-volumetry agent, a retrieval "
        "agent, and an ethics guardrail cross-check every prediction. A local Ollama-backed "
        "reasoner produces the final grounded narrative — entirely on free, on-device models."
    )
    rb.bullet(f"Cohen's kappa (chance-corrected agreement): {vision.get('kappa', 0):.3f}.")
    rb.bullet(
        f"Mean model confidence on evaluated slices: {vision.get('mean_confidence', 0) * 100:.1f}%."
    )
    rb.bullet(
        f"Ethics guardrail correctly adjudicated {int(guardrail.get('accuracy', 0) * guardrail.get('n', 0))}"
        f"/{guardrail.get('n', 0)} adversarial safety cases."
    )
    if not vision.get("trained", True):
        rb.paragraph(
            "NOTE: trained weights were not found; vision metrics reflect an UNTRAINED "
            "baseline and should be regenerated after training.",
            color=BAD,
        )

    # --- Vision metrics ---
    rb.new_page()
    rb.heading("2. Vision Agent Classification Performance", 1)
    rb.paragraph(
        f"Evaluated on {vision.get('n', 0)} slices from SUBJECT-DISJOINT held-out patients "
        "(no subject overlap with training), using the training-time validation transform "
        "(grayscale, 224x224, normalized). Held-out test subjects per class: "
        + ", ".join(
            f"{c}={vision.get('test_subjects', {}).get(i, 0)}"
            for i, c in enumerate(vision.get("classes", []))
        )
        + ". A class with very few held-out subjects yields a statistically meaningless "
        "per-class metric."
    )
    classes = vision.get("classes", CLASS_NAMES)
    rb.grouped_bar(
        "Per-class precision / recall / F1",
        categories=classes,
        series={
            "Precision": vision.get("precision", [0] * len(classes)),
            "Recall": vision.get("recall", [0] * len(classes)),
            "F1": vision.get("f1", [0] * len(classes)),
        },
        ymax=1.0,
    )
    rows = []
    for i, c in enumerate(classes):
        rows.append(
            (
                c,
                f"{vision['precision'][i]:.2f}",
                f"{vision['recall'][i]:.2f}",
                f"{vision['f1'][i]:.2f}",
                str(int(vision["support"][i])),
            )
        )
    rb.table(
        ["Class", "Precision", "Recall", "F1", "Support"], rows, col_w=[360, 160, 160, 160, 180]
    )

    rb.new_page()
    rb.heading("2.1 Confusion Matrix", 2)
    rb.heatmap("Confusion matrix (true vs predicted)", vision.get("cm"), classes, classes)
    rb.heading("2.2 Clinical screening view", 2)
    rb.paragraph(
        "Collapsing the four classes into a binary screening decision (any dementia vs "
        "non-demented) yields the operating point most relevant to triage:"
    )
    rb.bullet(
        f"Sensitivity (impaired correctly flagged): {vision.get('sensitivity', 0) * 100:.1f}%"
    )
    rb.bullet(f"Specificity (healthy correctly cleared): {vision.get('specificity', 0) * 100:.1f}%")

    # --- Cohort analysis ---
    rb.new_page()
    rb.heading("3. Cohort & Longitudinal Data Analysis", 1)
    if cohort.get("clinical_n"):
        rb.paragraph(
            f"Cross-sectional cohort: {cohort['clinical_n']} subjects. "
            f"Age {cohort['age_mean']:.1f}±{cohort['age_std']:.1f} yr; "
            f"MMSE {cohort['mmse_mean']:.1f}±{cohort['mmse_std']:.1f}; "
            f"mean nWBV {cohort['nwbv_mean']:.3f}."
        )
        if "sex" in cohort:
            rb.bullet(
                "Sex distribution: " + ", ".join(f"{k}={v}" for k, v in cohort["sex"].items())
            )
    abg = cohort.get("atrophy_by_group", {})
    if abg:
        rb.paragraph(
            "The temporal analyst computes whole-brain atrophy velocity (%/yr) from "
            "longitudinal nWBV. Group separation validates the longitudinal agent:"
        )
        labels = list(abg.keys())
        vals = [abg[g]["mean"] for g in labels]
        cols = []
        for g in labels:
            gl = g.lower()
            cols.append(
                BAD if "demente" in gl and "non" not in gl else (WARN if "convert" in gl else GOOD)
            )
        rb.hbar(
            "Mean atrophy velocity by diagnostic group (%/yr)",
            [f"{g} (n={abg[g]['n']})" for g in labels],
            vals,
            colors=cols,
            unit="%",
        )

    # --- Guardrail ---
    rb.new_page()
    rb.heading("4. Ethics Guardrail Effectiveness", 1)
    rb.paragraph(
        "The ethicist agent is stress-tested against a labelled battery of safe and "
        "unsafe diagnostic proposals. It must flag unsafe/contradictory cases and pass "
        "consistent ones."
    )
    rb.kpi_row(
        [
            ("Battery accuracy", f"{guardrail['accuracy'] * 100:.0f}%", GOOD),
            ("Cases tested", f"{guardrail['n']}", ACCENT),
        ]
    )
    rb.table(
        ["Scenario", "Guardrail", "Expected", "Result"],
        guardrail["rows"],
        col_w=[560, 160, 160, 140],
    )

    # --- Volumetry ---
    rb.new_page()
    rb.heading("5. Regional Volumetry (FreeSurfer aseg)", 1)
    rb.paragraph(
        "The regional-volumetry agent normalizes FreeSurfer subcortical volumes by eTIV "
        "and z-scores them against an elderly normative reference. Demonstration on a "
        "healthy vs AD-like segmentation shows clear separation in the medial-temporal "
        "risk score:"
    )
    vrows = []
    for label, r in volumetry.items():
        vrows.append(
            (label, f"{r['lh_z']:+.2f}", f"{r['vent_z']:+.2f}", f"{r['mta_risk']:.2f}", r["stage"])
        )
    rb.table(
        ["Subject", "Hippocampus z", "Ventricle z", "MTA risk", "Stage"],
        vrows,
        col_w=[300, 190, 190, 150, 190],
    )
    labels = list(volumetry.keys())
    rb.hbar(
        "Medial-temporal-atrophy risk score",
        labels,
        [volumetry[lab]["mta_risk"] for lab in labels],
        colors=[GOOD, BAD][: len(labels)],
        vmax=2.5,
    )

    # --- Edge benchmark ---
    rb.new_page()
    rb.heading("6. Edge Acceleration Benchmark (Snapdragon NPU)", 1)
    sizes = bench.get("sizes", {})
    if sizes.get("raw_mb"):
        rb.kpi_row(
            [
                ("FP32 model", f"{sizes['raw_mb']:.1f} MB", ACCENT),
                ("INT8 model", f"{sizes['int8_mb']:.1f} MB", ACCENT2),
                ("Size reduction", f"{sizes['reduction_pct']:.0f}%", GOOD),
            ]
        )
    provs = bench.get("providers", [])
    measured = [p for p in provs if p.get("ms")]
    if measured:
        rb.grouped_bar(
            "Per-inference latency by execution provider (ms, lower is better)",
            categories=[p["provider"] for p in measured],
            series={"Latency (ms)": [p["ms"] for p in measured]},
            ymax=max(p["ms"] for p in measured) * 1.25,
            fmt="{:.1f}",
        )
    rows = []
    for p in provs:
        rows.append(
            (
                p["provider"],
                f"{p['ms']:.1f} ms" if p.get("ms") else "n/a",
                "available" if p.get("ms") else p.get("error", "unavailable"),
            )
        )
    if bench.get("torch_ms"):
        rows.append(("PyTorch FP32 (reference)", f"{bench['torch_ms']:.1f} ms", "baseline"))
    rb.table(["Execution provider", "Latency", "Status"], rows, col_w=[460, 240, 360])
    rb.paragraph(
        "The pipeline auto-selects the fastest available provider via the "
        "QNN→DirectML→CPU fallback chain. On Snapdragon X hardware the QNN row maps to "
        "the Hexagon NPU; on other machines it gracefully degrades to DirectML or CPU.",
        color=ACCENT,
    )

    # --- Methodology ---
    rb.new_page()
    rb.heading("7. Methodology & Limitations", 1)
    rb.bullet(
        "Vision metrics use a SUBJECT-DISJOINT (patient-grouped) split - no subject appears "
        "in both training and test (src/pipeline/data_split.py). This is the honest measure of "
        "generalization; image-level splits inflate accuracy because there are ~240 "
        "near-identical slices per subject. Evaluated on CPU with training-consistent "
        "normalization (seed=42)."
    )
    rb.bullet(
        "Normative volumetry reference values are screening priors consolidated from "
        "FreeSurfer/ADNI-style literature, not site-specific ground truth."
    )
    rb.bullet(
        "Latency benchmarks are single-stream wall-clock means over 20 runs after warmup; "
        "absolute numbers vary with hardware and ONNX Runtime build."
    )
    rb.bullet(
        "The bundled OASIS-1 slices are 2D; full regional volumetry requires the OASIS-3/4 "
        "FreeSurfer derivatives fetched by scripts/oasis/."
    )
    rb.bullet(
        "This system is screening decision-support and is not a substitute for a "
        "neurologist's diagnosis; all flagged cases route to human review."
    )

    return rb.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate the OASIS effectiveness PDF report.")
    parser.add_argument("--max-per-class", type=int, default=60)
    parser.add_argument(
        "--weights", type=str, default=None, help="path to model weights (.pth); default = bundled"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(_ROOT, "docs", "OASIS_Pipeline_Effectiveness_Analysis.pdf"),
    )
    args = parser.parse_args()

    print("[1/5] Evaluating vision agent on subject-disjoint HELD-OUT patients ...")
    vision = evaluate_vision(_ROOT, args.max_per_class, weights_path=args.weights)
    print(
        f"      accuracy={vision['accuracy'] * 100:.1f}%  balanced={vision['balanced_accuracy'] * 100:.1f}%"
    )
    print("[2/5] Analyzing cohort + longitudinal data ...")
    cohort = analyze_cohort(_ROOT)
    print("[3/5] Stress-testing ethics guardrail ...")
    guardrail = evaluate_guardrail(_ROOT)
    print("[4/5] Demonstrating regional volumetry ...")
    volumetry = evaluate_volumetry(_ROOT)
    print("[5/5] Benchmarking edge inference providers ...")
    bench = benchmark_inference(_ROOT)

    out = build_report(_ROOT, vision, cohort, guardrail, volumetry, bench, args.out)
    print(f"\n[SUCCESS] Report written to: {out}")


if __name__ == "__main__":
    main()
