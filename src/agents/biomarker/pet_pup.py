"""
OASIS-3 / OASIS-4 PET Unified Pipeline (PUP) SUVR ingestion.

Reads the regional SUVR tables produced by PUP (downloaded via
``scripts/oasis/download_oasis_pup_tar.sh``) and derives the two PET-based axes
of the ATN framework:

    A (Amyloid)  — mean cortical SUVR (MCSUVR) from the standard summary regions
                   (precuneus, prefrontal, gyrus rectus / medial-orbitofrontal,
                   lateral temporal), referenced to cerebellum -> Centiloid.
    T (Tau)      — a temporal/Braak meta-ROI SUVR (entorhinal, amygdala,
                   fusiform, inferior/middle temporal) for tau tracers (AV1451).

The tracer (and hence whether a PUP session is amyloid or tau) is inferred from
the PUP id / folder name, e.g. ``OAS30001_AV45_PUPTIMECOURSE_d2430`` (amyloid,
florbetapir), ``..._PIB_...`` (amyloid, Pittsburgh compound B),
``..._AV1451_...`` (tau, flortaucipir).

PUP output layouts vary, so the parser is format-tolerant: it accepts long
``ROI,SUVR`` tables, wide single-row tables (one column per ROI), and simple
``key value`` text, matching ROI names case-insensitively against the summary
region sets below. When no PUP data is present the A/T axes simply stay
``indeterminate`` and the pipeline falls back to neurodegeneration-only staging.
"""

from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Tracer detection from PUP id / filename.
AMYLOID_TRACERS = {
    "av45": "AV45",
    "fbp": "AV45",
    "florbetapir": "AV45",
    "pib": "PIB",
    "fbb": "FBB",
    "florbetaben": "FBB",
    "nav": "AV45",
}
TAU_TRACERS = {"av1451": "AV1451", "flortaucipir": "AV1451", "mk6240": "MK6240", "tau": "AV1451"}

# Cortical amyloid-summary ROIs (FreeSurfer aparc substrings).
CORTICAL_SUMMARY = [
    "precuneus",
    "superiorfrontal",
    "rostralmiddlefrontal",
    "caudalmiddlefrontal",
    "lateralorbitofrontal",
    "medialorbitofrontal",  # ~ gyrus rectus
    "superiortemporal",
    "middletemporal",
    "inferiortemporal",
]
# Tau temporal/Braak meta-ROI ROIs.
TAU_METAROI = ["entorhinal", "amygdala", "fusiform", "inferiortemporal", "middletemporal"]

# Tracer-specific SUVR -> Centiloid (slope, intercept); pipeline-dependent priors.
CENTILOID = {"PIB": (93.7, -94.6), "AV45": (196.9, -196.0), "FBB": (153.4, -154.9)}


@dataclass
class PUPResult:
    subject_id: str
    amyloid_suvr: Optional[float] = None
    amyloid_centiloid: Optional[float] = None
    amyloid_tracer: Optional[str] = None
    amyloid_positive: Optional[bool] = None
    tau_suvr: Optional[float] = None
    tau_tracer: Optional[str] = None
    tau_positive: Optional[bool] = None
    sources: List[str] = field(default_factory=list)
    summary: str = "No PET (PUP) data found."

    def to_dict(self) -> Dict[str, object]:
        def r(x):
            return None if x is None else round(x, 3)

        return {
            "subject_id": self.subject_id,
            "amyloid_suvr": r(self.amyloid_suvr),
            "amyloid_centiloid": None
            if self.amyloid_centiloid is None
            else round(self.amyloid_centiloid, 1),
            "amyloid_tracer": self.amyloid_tracer,
            "amyloid_positive": self.amyloid_positive,
            "tau_suvr": r(self.tau_suvr),
            "tau_tracer": self.tau_tracer,
            "tau_positive": self.tau_positive,
            "sources": self.sources,
            "summary": self.summary,
        }


class PUPPetParser:
    """Parses PUP regional SUVR outputs into amyloid/tau summaries."""

    def __init__(
        self,
        pup_root: Optional[str] = None,
        amyloid_threshold_cl: float = 20.0,
        tau_threshold_suvr: float = 1.27,
    ):
        self.pup_root = pup_root
        self.amyloid_threshold_cl = amyloid_threshold_cl
        self.tau_threshold_suvr = tau_threshold_suvr
        if pup_root and os.path.isdir(pup_root):
            print(f"[+] PUP PET parser bound to: {pup_root}")
        else:
            print("[*] PUP PET parser ready (no PET root yet; ATN A/T stay indeterminate).")

    # ------------------------------------------------------------- tracers
    @staticmethod
    def detect_tracer(name: str):
        low = name.lower()
        for key, tr in TAU_TRACERS.items():
            if key in low:
                return ("tau", tr)
        for key, tr in AMYLOID_TRACERS.items():
            if key in low:
                return ("amyloid", tr)
        return (None, None)

    @staticmethod
    def suvr_to_centiloid(suvr: float, tracer: str) -> float:
        slope, intercept = CENTILOID.get(tracer.upper(), CENTILOID["PIB"])
        return slope * suvr + intercept

    # ------------------------------------------------------------- parsing
    @staticmethod
    def _parse_roi_suvr(path: str) -> Dict[str, float]:
        """Best-effort ROI->SUVR extraction from long/wide/key-value tables."""
        roi_suvr: Dict[str, float] = {}
        try:
            with open(path, "r", errors="ignore", newline="") as fh:
                sample = fh.read()
        except OSError:
            return roi_suvr
        lines = [ln for ln in sample.splitlines() if ln.strip()]
        if not lines:
            return roi_suvr

        delim = "," if "," in lines[0] else ("\t" if "\t" in lines[0] else None)
        rows = list(csv.reader(lines, delimiter=delim)) if delim else [ln.split() for ln in lines]
        if not rows:
            return roi_suvr
        header = [c.strip().lower() for c in rows[0]]

        def to_float(x):
            try:
                return float(str(x).strip())
            except (TypeError, ValueError):
                return None

        # Long format: a region column + a suvr/value column.
        roi_col = next(
            (i for i, h in enumerate(header) if h in ("roi", "region", "label", "structure")), None
        )
        val_col = next(
            (i for i, h in enumerate(header) if "suvr" in h or h in ("value", "mean", "fsuvr")),
            None,
        )
        if roi_col is not None and val_col is not None and roi_col != val_col:
            for row in rows[1:]:
                if len(row) > max(roi_col, val_col):
                    v = to_float(row[val_col])
                    if v is not None:
                        roi_suvr[str(row[roi_col]).strip().lower()] = v
            if roi_suvr:
                return roi_suvr

        # Wide format: header = ROI names, single (or first) data row = SUVR values.
        if len(rows) >= 2:
            for i, h in enumerate(header):
                if i < len(rows[1]):
                    v = to_float(rows[1][i])
                    if v is not None and 0.2 < v < 6.0:  # plausible SUVR range
                        roi_suvr[h] = v
            if roi_suvr:
                return roi_suvr

        # key value text
        for ln in lines:
            m = re.match(r"\s*([A-Za-z][\w\-\.]+)[\s=:]+([-+]?\d*\.?\d+)\s*$", ln)
            if m:
                roi_suvr[m.group(1).lower()] = float(m.group(2))
        return roi_suvr

    @staticmethod
    def _mean_matching(roi_suvr: Dict[str, float], patterns: List[str]) -> Optional[float]:
        vals = [v for roi, v in roi_suvr.items() if any(p in roi for p in patterns)]
        return sum(vals) / len(vals) if vals else None

    def parse_pup_session(self, path: str) -> Optional[Dict[str, object]]:
        """Parse one PUP output (dir or file). Returns a partial measure dict."""
        name = os.path.basename(path.rstrip("/\\"))
        modality, tracer = self.detect_tracer(name)
        files: List[str] = []
        if os.path.isdir(path):
            for pat in ("*.csv", "*.txt", "*.suvr", "*SUVR*", "*suvr*"):
                files += glob.glob(os.path.join(path, "**", pat), recursive=True)
        elif os.path.isfile(path):
            files = [path]
        roi_suvr: Dict[str, float] = {}
        used: List[str] = []
        for f in sorted(set(files)):
            parsed = self._parse_roi_suvr(f)
            if parsed:
                roi_suvr.update(parsed)
                used.append(os.path.basename(f))
        if not roi_suvr:
            return None
        if modality is None:
            # Heuristic: presence of explicit summary/tot key.
            modality = "amyloid"
            tracer = tracer or "PIB"
        result = {"modality": modality, "tracer": tracer, "files": used}
        if modality == "amyloid":
            mc = roi_suvr.get("mcsuvr") or roi_suvr.get("cortmean") or roi_suvr.get("tot_cortmean")
            result["suvr"] = (
                mc if mc is not None else self._mean_matching(roi_suvr, CORTICAL_SUMMARY)
            )
        else:
            result["suvr"] = self._mean_matching(roi_suvr, TAU_METAROI)
        return result if result.get("suvr") is not None else None

    # ------------------------------------------------------------- subject
    def analyze_subject(self, subject_id: str) -> PUPResult:
        res = PUPResult(subject_id=subject_id)
        if not self.pup_root or not os.path.isdir(self.pup_root):
            return res
        subj = (subject_id or "").split("_")[0].lower()
        sessions = [
            d
            for d in glob.glob(os.path.join(self.pup_root, "*"))
            if subj in os.path.basename(d).lower()
        ]
        for sess in sessions:
            parsed = self.parse_pup_session(sess)
            if not parsed:
                continue
            res.sources.append(os.path.basename(sess))
            if parsed["modality"] == "amyloid" and res.amyloid_suvr is None:
                res.amyloid_suvr = float(parsed["suvr"])
                res.amyloid_tracer = parsed["tracer"]
                res.amyloid_centiloid = self.suvr_to_centiloid(
                    res.amyloid_suvr, res.amyloid_tracer or "PIB"
                )
                res.amyloid_positive = res.amyloid_centiloid >= self.amyloid_threshold_cl
            elif parsed["modality"] == "tau" and res.tau_suvr is None:
                res.tau_suvr = float(parsed["suvr"])
                res.tau_tracer = parsed["tracer"]
                res.tau_positive = res.tau_suvr >= self.tau_threshold_suvr
        res.summary = self._summary(res)
        return res

    @staticmethod
    def _summary(res: "PUPResult") -> str:
        parts = []
        if res.amyloid_suvr is not None:
            parts.append(
                f"amyloid {res.amyloid_tracer} SUVR {res.amyloid_suvr:.2f} "
                f"({res.amyloid_centiloid:.0f} CL, {'+' if res.amyloid_positive else '-'})"
            )
        if res.tau_suvr is not None:
            parts.append(
                f"tau {res.tau_tracer} meta-ROI SUVR {res.tau_suvr:.2f} ({'+' if res.tau_positive else '-'})"
            )
        return "PET (PUP): " + "; ".join(parts) if parts else "No PET (PUP) data found."


if __name__ == "__main__":
    import tempfile

    # Synthetic amyloid (florbetapir, positive) + tau (AV1451, positive) PUP CSVs.
    amyloid_csv = "ROI,SUVR\nPrecuneus,1.62\nSuperiorFrontal,1.55\nMedialOrbitofrontal,1.48\nMiddleTemporal,1.50\nCerebellum,1.00\n"
    tau_csv = "region,fsuvr\nEntorhinal,1.55\nAmygdala,1.40\nFusiform,1.32\nInferiorTemporal,1.38\n"

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "OAS30001_AV45_PUPTIMECOURSE_d2430"))
    os.makedirs(os.path.join(root, "OAS30001_AV1451_PUPTIMECOURSE_d2500"))
    with open(os.path.join(root, "OAS30001_AV45_PUPTIMECOURSE_d2430", "suvr.csv"), "w") as f:
        f.write(amyloid_csv)
    with open(os.path.join(root, "OAS30001_AV1451_PUPTIMECOURSE_d2500", "suvr.csv"), "w") as f:
        f.write(tau_csv)

    parser = PUPPetParser(pup_root=root)
    out = parser.analyze_subject("OAS30001")
    print("\n--- PUP PET ingestion self-test ---")
    print(out.summary)
    print(out.to_dict())
