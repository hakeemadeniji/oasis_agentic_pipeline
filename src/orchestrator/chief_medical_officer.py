import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Synchronize directory structure across application layers
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(src_dir)

from agents.vision.vision_agent import AlzheimerVisionAgent
from agents.vision.explainer_agent import RadiomicsExplainerAgent
from agents.biomarker.biomarker_agent import ClinicalBiomarkerAgent
from agents.biomarker.temporal_analyst import TemporalAnalystAgent
from agents.biomarker.volumetry_agent import RegionalVolumetryAgent
from agents.biomarker.atn_classifier import ATNBiomarkerProfiler
from agents.biomarker.pet_pup import PUPPetParser
from agents.rag.rag_agent import MedicalLibrarianAgent
from agents.llm.llm_reasoner import ClinicalReasonerAgent
from agents.llm.differential_agent import DifferentialDiagnosisAgent
from agents.llm.therapeutic_agent import TherapeuticInsightAgent
from pipeline.research.cure_research import CureResearchEngine
from orchestrator.ethicist_agent import MedicalEthicistAgent
from config import get_settings


class AdvancedChiefMedicalOfficer:
    """
    Advanced Heterogeneous Swarm Orchestrator (hybrid edge-cloud).
    Directs independent clinical agents spanning spatial imaging, longitudinal
    kinetics, regional volumetry, text RAG, compliance guardrails, and an
    Ollama-backed clinical reasoner -- with NPU-accelerated inference and zero
    paid API tokens.
    """

    def __init__(self, workspace_root: str):
        print("\n=======================================================")
        print("  [SYSTEM BOOT] Advanced Multi-Agent Swarm Connected  ")
        print("=======================================================\n")

        self.workspace_root = workspace_root
        self.settings = get_settings()
        print(f"[CONFIG] {self.settings.summary()}")
        self.device = torch.device(self.settings.resolve_device())

        # Establish accurate data file maps
        self.cross_csv = os.path.join(
            self.workspace_root, "data", "oasis_raw", "oasis_clinical_data.csv"
        )
        self.long_csv = os.path.join(
            self.workspace_root, "data", "oasis_raw", "oasis_longitudinal.csv"
        )

        # 1. Initialize Baseline Tabular Pipelines
        self.biomarker_agent = ClinicalBiomarkerAgent()
        self.patient_tensors, self.patient_df = self.biomarker_agent.ingest_and_process(
            self.cross_csv
        )

        # 2. Initialize Agent 5: Temporal Analyst
        self.temporal_agent = TemporalAnalystAgent(self.long_csv)

        # 3. Initialize Agent 3: Medical Librarian RAG
        self.rag_agent = MedicalLibrarianAgent()
        self.rag_agent.ingest_medical_guidelines(
            [
                "Normal Cognition presents with an MMSE from 26 to 30. Structural brain mass is well maintained.",
                "Very Mild Dementia features slight memory changes, with an MMSE score typically hanging between 24 and 26.",
                "Mild Dementia shows progressive confusion, tracking with an MMSE score down between 20 and 24.",
                "Moderate Dementia presents severe memory degradation, tracking with an MMSE score between 13 and 20.",
            ]
        )

        # 4. Initialize Agent 1: Vision Architecture
        # IMPORTANT: order must match torchvision ImageFolder's alphabetical class
        # indexing used during training, otherwise correct predictions get mislabeled.
        self.class_names = [
            "Mild Dementia",
            "Moderate Dementia",
            "Non Demented",
            "Very mild Dementia",
        ]
        self.vision_agent = AlzheimerVisionAgent(num_classes=len(self.class_names)).to(self.device)

        weights_path = os.path.join(
            self.workspace_root, "src", "pipeline", "onnx_inference", "best_vision_agent.pth"
        )
        if os.path.exists(weights_path):
            self.vision_agent.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("[+] Vision Agent loaded with optimized training weights.")
        else:
            print(
                "[!] Vision weights training in separate window. Initializing untrained framework for routing tests."
            )

        # 5. Initialize Agent 4: Radiomics Explainer (Grad-CAM)
        self.explainer_agent = RadiomicsExplainerAgent(self.vision_agent)

        # 6. Initialize Agent 6: Medical Ethicist Guardrail
        self.ethicist_agent = MedicalEthicistAgent(confidence_floor=self.settings.confidence_floor)

        # 7. Initialize Agent 9: Regional Volumetry Analyst (FreeSurfer aseg)
        fs_root = self.settings.freesurfer_root
        if not os.path.isabs(fs_root):
            fs_root = os.path.join(self.workspace_root, fs_root)
        self.volumetry_agent = RegionalVolumetryAgent(freesurfer_root=fs_root)

        # 8. Initialize Agent 8: Hybrid Clinical Reasoner (Ollama + Claude)
        self.reasoner_agent = ClinicalReasonerAgent()

        # 11/12. Differential Diagnosis (Claude) + Therapeutic Insight + cure-research engine
        self.differential_agent = DifferentialDiagnosisAgent()
        self.therapeutic_agent = TherapeuticInsightAgent()
        self.research_engine = CureResearchEngine(self.workspace_root)
        self._research_cache = None

        # 9. Initialize Agent 10: ATN Biomarker Profiler (NIA-AA framework) + PUP PET ingestion
        self.atn_profiler = ATNBiomarkerProfiler(
            amyloid_threshold_cl=self.settings.amyloid_positive_centiloid,
            tau_threshold_suvr=self.settings.tau_positive_suvr,
        )
        pup_root = self.settings.pup_root
        if not os.path.isabs(pup_root):
            pup_root = os.path.join(self.workspace_root, pup_root)
        self.pet_pup = PUPPetParser(
            pup_root=pup_root,
            amyloid_threshold_cl=self.settings.amyloid_positive_centiloid,
            tau_threshold_suvr=self.settings.tau_positive_suvr,
        )

        # NOTE: must match the training-time normalization (mean=0.5, std=0.5);
        # omitting it feeds out-of-distribution inputs and degrades accuracy.
        self.image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        print("\n[SUCCESS] Hexagonal Multi-Agent Architecture Fully Linked.\n")

    def execute_comprehensive_diagnosis(
        self, patient_idx: int, image_path: str, mock_subject_id: str = "OAS2_0001"
    ):
        """Executes full diagnostic pass across all 6 specialized entities."""
        print(f"--- Initiating Diagnostic Execution Loop: Patient Index {patient_idx} ---")

        # A. Gather Clinical Tabular Profiling
        patient_row = self.patient_df.iloc[patient_idx]
        mmse = float(patient_row["MMSE"])
        age = float(patient_row["Age"])

        # B. Gather Longitudinal Kinetics
        long_metrics = self.temporal_agent.calculate_progression_trajectory(mock_subject_id)
        atrophy_vel = float(long_metrics.get("atrophy_velocity_pct", 0.0))

        # B2. Regional Volumetry (Agent 9): prefer real FreeSurfer aseg.stats,
        # otherwise estimate from whole-brain biomarkers so the agent always returns data.
        volumetry = self.volumetry_agent.analyze_subject(mock_subject_id)
        if volumetry.source == "unavailable":
            etiv = float(patient_row.get("eTIV", 1500.0))
            # OASIS eTIV is reported in cm^3 in the tabular CSV; convert to mm^3.
            etiv_mm3 = etiv * 1000.0 if etiv < 10000 else etiv
            nwbv = float(patient_row.get("nWBV", 0.73))
            volumetry = self.volumetry_agent.estimate_from_biomarkers(
                mock_subject_id, etiv_mm3, nwbv
            )

        # B3. ATN biomarker profiling (Agent 10). The A/T axes are driven by real
        # OASIS-3 PET (PUP) SUVR when available; the N axis comes from volumetry.
        pet = self.pet_pup.analyze_subject(mock_subject_id)
        hippo_zs = [r.z_score for r in volumetry.regions if "Hippocampus" in r.structure]
        atn = self.atn_profiler.classify(
            amyloid_suvr=pet.amyloid_suvr,
            amyloid_centiloid=pet.amyloid_centiloid,
            amyloid_tracer=pet.amyloid_tracer or "PIB",
            tau_suvr=pet.tau_suvr,
            hippocampus_z=(sum(hippo_zs) / len(hippo_zs)) if hippo_zs else None,
            mta_risk=volumetry.mta_risk_score,
            nwbv=float(patient_row.get("nWBV", 0.73)),
        )

        # C. Gather Vision Analytics
        img = Image.open(image_path).convert("L")
        raw_transformed = self.image_transform(img)
        assert isinstance(raw_transformed, torch.Tensor)
        img_tensor = raw_transformed.unsqueeze(0).to(self.device)

        # Enable gradient tracking for Grad-CAM processing
        img_tensor.requires_grad = True

        vision_output = self.vision_agent(img_tensor)
        probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
        pred_idx = int(torch.argmax(probabilities).item())
        pred_class = self.class_names[pred_idx]
        confidence = float(probabilities[pred_idx].item() * 100)

        # D. Execute Explainable AI Analysis
        heatmap = self.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
        peak_structural_focus = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # E. Enforce Compliance & Ethical Guardrails
        is_flagged, restriction_log = self.ethicist_agent.audit_diagnostic_proposal(
            predicted_class=pred_class,
            confidence=confidence,
            mmse_score=mmse,
            atrophy_velocity=atrophy_vel,
        )

        # F. Trigger Semantic Literature Search
        query = f"Clinical metrics for an MMSE of {mmse:.1f} and localized brain tissue changes."
        rag_results = self.rag_agent.query(query, top_k=2)
        rag_output = rag_results[0][0]

        # F2. Hybrid Edge-Cloud Clinical Reasoner (Agent 8): synthesize a grounded
        # natural-language summary from every agent's structured output via Ollama.
        authorized_class = pred_class if not is_flagged else "DIAGNOSIS WITHHELD (human review)"
        evidence = {
            "prediction": pred_class,
            "authorized_class": authorized_class,
            "confidence": confidence,
            "age": round(age, 1),
            "mmse": round(mmse, 1),
            "clinical_trend": long_metrics.get("clinical_trend", "N/A"),
            "atrophy_velocity": atrophy_vel,
            "volumetry_summary": volumetry.summary,
            "atn_profile": atn.profile,
            "atn_category": atn.category,
            "atn_a": atn.a_status,
            "atn_t": atn.t_status,
            "atn_n": atn.n_status,
            "hippocampus_z": (sum(hippo_zs) / len(hippo_zs)) if hippo_zs else None,
            "ethics_flagged": is_flagged,
            "ethics_message": restriction_log,
            "rag_context": [doc for doc, _ in rag_results],
        }
        reasoning = self.reasoner_agent.synthesize(evidence)
        # Agent 11: ranked differential across dementia etiologies (Claude deep tier).
        differential = self.differential_agent.analyze(evidence)

        # G. Build Consolidated Diagnostic Output
        print("\n=======================================================")
        print("         SWARM CONSOLIDATED CLINICAL DIAGNOSIS         ")
        print("=======================================================")
        print(
            f"Patient Tracking ID :  {patient_row['Subject_ID']} (Longitudinal Profile: {mock_subject_id})"
        )
        print(f"Chronological Age   :  {age:.1f} years")
        print(f"Cognitive Metrics   :  MMSE {mmse:.1f}/30.0")
        print("-------------------------------------------------------")
        print("LONGITUDINAL KINETICS (Agent 5):")
        print(
            f" > Monitored Timeline: {long_metrics.get('years_monitored', 'N/A')} years across {long_metrics.get('visits_tracked', 'N/A')} clinical visits."
        )
        print(f" > Atrophy Velocity  : {atrophy_vel:.3f}% whole brain structural loss / year.")
        print(f" > Progression Trend : {long_metrics.get('clinical_trend', 'N/A')}")
        print("-------------------------------------------------------")
        print("EXPLAINABLE RADIOMICS (Agents 1 & 4):")
        print(f" > Initial Prediction: {pred_class} (Model Confidence: {confidence:.2f}%)")
        print(
            f" > Grad-CAM Heatmap  : Localized peak feature density mapped to pixel tensor indices {peak_structural_focus}."
        )
        print("-------------------------------------------------------")
        print("COMPLIANCE & RISK MANAGEMENT REPORT (Agent 6):")
        if is_flagged:
            print(" [CRITICAL ALARM] DIAGNOSIS OVERRIDDEN BY ETHICIST AGENT")
            print(
                " > Action Taken      : Halted deployment. Routed case file to Human Neurologist."
            )
            print(f" > Infraction Reason : {restriction_log}")
        else:
            print(f" [STATUS: VERIFIED] {restriction_log}")
            print(f" > Authorized Diagnostic Classification: {pred_class}")
        print("-------------------------------------------------------")
        print("REGIONAL VOLUMETRY (Agent 9):")
        print(f" > Source            : {volumetry.source} | MTA stage: {volumetry.mta_stage}")
        print(f" > {volumetry.summary}")
        if volumetry.flags:
            for flag in volumetry.flags:
                print(f"   - {flag}")
        print("-------------------------------------------------------")
        print("ATN BIOMARKER PROFILE (Agent 10):")
        print(f" > Profile {atn.profile} | {atn.category}")
        print(f" > {pet.summary}")
        print(f" > {atn.summary}")
        print("-------------------------------------------------------")
        print(f"DIFFERENTIAL DIAGNOSIS (Agent 11) [{differential.provider}]:")
        for rank in differential.ranking[:3]:
            print(f" > {rank.get('likelihood', '?'):>3}%  {rank.get('etiology', '?')}")
        print(f" > Work-up: {'; '.join(differential.recommended_workup[:2])}")
        print("-------------------------------------------------------")
        print("SUPPORTING MEDICAL LITERATURE (Agent 3):")
        print(f" > {rag_output}")
        print("-------------------------------------------------------")
        print(
            f"HYBRID EDGE-CLOUD CLINICAL REASONER (Agent 8) [tier={reasoning.tier}, model={reasoning.model}]:"
        )
        print(f" > {reasoning.narrative}")
        print("=======================================================\n")

    def run_cure_research(self) -> dict:
        """
        Run the deterministic cure-research experiment engine (Agent: research) and
        the Claude-powered Therapeutic Insight agent (Agent 12) over the cohort,
        returning grounded, testable hypotheses. Cached after first run.
        """
        if self._research_cache is None:
            report = self.research_engine.run().to_dict()
            insight = self.therapeutic_agent.analyze(report)
            self._research_cache = {"report": report, "insight": insight.to_dict()}
        return self._research_cache


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmo = AdvancedChiefMedicalOfficer(ROOT)

    # Locate a real dataset slice for the tracking validation test
    test_folder = os.path.join(ROOT, "data", "oasis_raw", "Non Demented")
    try:
        sample_img = [f for f in os.listdir(test_folder) if f.endswith((".jpg", ".jpeg", ".png"))][
            0
        ]
        full_img_path = os.path.join(test_folder, sample_img)

        cmo.execute_comprehensive_diagnosis(patient_idx=0, image_path=full_img_path)
    except IndexError:
        print(f"[!] Target validation scan missing from directory path: {test_folder}")
