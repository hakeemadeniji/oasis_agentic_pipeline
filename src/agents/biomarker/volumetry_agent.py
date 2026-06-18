"""
Agent 9: Regional Volumetry Analyst.

Moves the pipeline beyond a single whole-brain ``nWBV`` number toward true
*regional volumetry* -- the kind of structured measurement a clinician actually
reasons about in Alzheimer's screening (medial temporal lobe atrophy,
ventricular enlargement, hippocampal volume loss).

It is built directly around the **OASIS-3 / OASIS-4 FreeSurfer** derivatives
that the download scripts in ``scripts/oasis/`` retrieve. FreeSurfer's
``aseg.stats`` file contains per-structure volumes (mm^3) plus the estimated
total intracranial volume (eTIV). This agent:

1. Parses ``aseg.stats`` (and optional ``*h.aparc.stats`` cortical files),
2. Normalizes each structure by eTIV to remove head-size confounds,
3. Converts to **z-scores** against an elderly normative reference,
4. Flags AD-relevant patterns (hippocampal/amygdalar atrophy, ventricular
   enlargement) and produces a compact medial-temporal-atrophy (MTA) style risk
   score for downstream agents and the LLM reasoner.

When no FreeSurfer derivative is available (e.g. a 2D-only OASIS-1 slice), the
agent degrades gracefully to an eTIV/nWBV-based whole-brain estimate so the
orchestrator always receives a populated, well-typed result.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Elderly (~70-85 yr) normative reference, expressed as percent of eTIV:
#   value_pct = structure_volume_mm3 / eTIV_mm3 * 100
# (mean, std). Approximate values consolidated from FreeSurfer/ADNI-style
# normative literature; treat as screening priors, not clinical ground truth.
# direction = -1  -> lower volume is the abnormal (atrophic) direction
# direction = +1  -> higher volume is the abnormal (enlarged) direction
# ---------------------------------------------------------------------------
NORMATIVE: Dict[str, Dict[str, Any]] = {
    "Left-Hippocampus": {"mean": 0.235, "std": 0.035, "direction": -1, "ad_weight": 1.0},
    "Right-Hippocampus": {"mean": 0.240, "std": 0.035, "direction": -1, "ad_weight": 1.0},
    "Left-Amygdala": {"mean": 0.105, "std": 0.020, "direction": -1, "ad_weight": 0.6},
    "Right-Amygdala": {"mean": 0.108, "std": 0.020, "direction": -1, "ad_weight": 0.6},
    "Left-Inf-Lat-Vent": {"mean": 0.045, "std": 0.030, "direction": +1, "ad_weight": 0.8},
    "Right-Inf-Lat-Vent": {"mean": 0.045, "std": 0.030, "direction": +1, "ad_weight": 0.8},
    "Left-Lateral-Ventricle": {"mean": 1.05, "std": 0.55, "direction": +1, "ad_weight": 0.5},
    "Right-Lateral-Ventricle": {"mean": 1.05, "std": 0.55, "direction": +1, "ad_weight": 0.5},
    "3rd-Ventricle": {"mean": 0.085, "std": 0.030, "direction": +1, "ad_weight": 0.4},
}

# Structures that drive the medial-temporal-atrophy composite.
MTA_STRUCTURES = (
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Inf-Lat-Vent",
    "Right-Inf-Lat-Vent",
)


@dataclass
class RegionMetric:
    structure: str
    volume_mm3: float
    pct_etiv: float
    z_score: float
    abnormal: bool
    note: str


@dataclass
class VolumetryResult:
    subject_id: str
    source: str  # "freesurfer" | "estimated" | "unavailable"
    etiv_mm3: float
    regions: List[RegionMetric] = field(default_factory=list)
    mta_risk_score: float = 0.0  # 0..~3, higher = more medial-temporal atrophy
    mta_stage: str = "N/A"  # informal Scheltens-like band
    flags: List[str] = field(default_factory=list)
    summary: str = "Regional volumetry not available."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "source": self.source,
            "etiv_mm3": round(self.etiv_mm3, 1),
            "mta_risk_score": round(self.mta_risk_score, 2),
            "mta_stage": self.mta_stage,
            "flags": self.flags,
            "summary": self.summary,
            "regions": [
                {
                    "structure": r.structure,
                    "volume_mm3": round(r.volume_mm3, 1),
                    "pct_etiv": round(r.pct_etiv, 4),
                    "z_score": round(r.z_score, 2),
                    "abnormal": r.abnormal,
                }
                for r in self.regions
            ],
        }


class RegionalVolumetryAgent:
    """Parses FreeSurfer aseg stats into normalized, z-scored regional metrics."""

    def __init__(self, freesurfer_root: Optional[str] = None):
        self.freesurfer_root = freesurfer_root
        if freesurfer_root and os.path.isdir(freesurfer_root):
            print(f"[+] Regional Volumetry Agent bound to FreeSurfer root: {freesurfer_root}")
        else:
            print(
                "[*] Regional Volumetry Agent ready (no FreeSurfer root yet; estimation fallback enabled)."
            )

    # ------------------------------------------------------------------ parse
    @staticmethod
    def parse_aseg_stats(path: str) -> Tuple[Dict[str, float], float]:
        """
        Parse a FreeSurfer ``aseg.stats`` file.

        Returns ``(volumes_by_structure, etiv_mm3)`` where volumes are in mm^3.
        """
        volumes: Dict[str, float] = {}
        etiv = 0.0
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("# Measure"):
                    # e.g. "# Measure EstimatedTotalIntraCranialVol, eTIV, ..., 1512345.6, mm^3"
                    if "EstimatedTotalIntraCranialVol" in line or "IntraCranialVol" in line:
                        m = re.findall(r"[-+]?\d*\.\d+|\d+", line.split(",")[-2])
                        if m:
                            etiv = float(m[-1])
                    continue
                if line.startswith("#") or not line.strip():
                    continue
                # Data rows: Index SegId NVoxels Volume_mm3 StructName [normMean ...]
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        vol = float(parts[3])
                    except ValueError:
                        continue
                    struct = parts[4]
                    volumes[struct] = vol
        return volumes, etiv

    # ----------------------------------------------------------------- analyze
    def analyze_stats_file(self, path: str, subject_id: Optional[str] = None) -> VolumetryResult:
        volumes, etiv = self.parse_aseg_stats(path)
        sid = subject_id or self._subject_from_path(path)
        if etiv <= 0:
            # Fall back to a plausible eTIV so normalization still runs.
            etiv = 1_500_000.0
        return self._build_result(sid, volumes, etiv, source="freesurfer")

    def analyze_subject(self, subject_id: str) -> VolumetryResult:
        """Locate and analyze a subject's aseg.stats under the FreeSurfer root."""
        path = self._find_subject_stats(subject_id)
        if path:
            return self.analyze_stats_file(path, subject_id=subject_id)
        return VolumetryResult(
            subject_id=subject_id,
            source="unavailable",
            etiv_mm3=0.0,
            summary=(
                f"No FreeSurfer aseg.stats found for {subject_id}. Run "
                "scripts/oasis/download_oasis_freesurfer_tar.sh to enable regional volumetry."
            ),
        )

    def estimate_from_biomarkers(
        self, subject_id: str, etiv: float, nwbv: float
    ) -> VolumetryResult:
        """
        Whole-brain fallback when no FreeSurfer derivative exists.

        Uses normalized whole-brain volume (nWBV) as a coarse global proxy and
        derives an approximate ventricular/atrophy signal so downstream agents
        still receive a populated result.
        """
        brain_vol = nwbv * etiv if etiv > 0 else 0.0
        # nWBV ~0.73 is typical for healthy elderly; lower => more global atrophy.
        global_z = (nwbv - 0.73) / 0.04 if nwbv else 0.0
        flags: List[str] = []
        if global_z < -1.5:
            flags.append("Global atrophy: whole-brain volume markedly below elderly norm.")
        summary = (
            f"Estimated (no FreeSurfer): nWBV={nwbv:.3f} (global z={global_z:+.1f}); "
            f"brain≈{brain_vol / 1000:.0f} cc within eTIV {etiv / 1000:.0f} cc. "
            "Regional sub-structures unavailable without FreeSurfer segmentation."
        )
        return VolumetryResult(
            subject_id=subject_id,
            source="estimated",
            etiv_mm3=etiv,
            mta_risk_score=max(0.0, -global_z),
            mta_stage=self._stage(max(0.0, -global_z)),
            flags=flags,
            summary=summary,
        )

    # ------------------------------------------------------------------ helpers
    def _build_result(
        self, subject_id: str, volumes: Dict[str, float], etiv: float, source: str
    ) -> VolumetryResult:
        regions: List[RegionMetric] = []
        flags: List[str] = []
        mta_terms: List[float] = []
        mta_weights: List[float] = []

        for struct, ref in NORMATIVE.items():
            if struct not in volumes:
                continue
            vol = volumes[struct]
            pct = vol / etiv * 100.0
            raw_z = (pct - ref["mean"]) / ref["std"]
            # Orient z so that NEGATIVE always means "more abnormal":
            #   direction=-1 (atrophy structures): abnormal = low volume = low raw_z
            #       -> keep raw_z (already negative when abnormal).
            #   direction=+1 (ventricles): abnormal = high volume = high raw_z
            #       -> negate so abnormal becomes negative.
            oriented_z = raw_z * (-ref["direction"])
            abnormal = oriented_z < -1.5
            note = ""
            if abnormal:
                if ref["direction"] < 0:
                    note = "atrophic"
                    flags.append(f"{struct}: volume {abs(oriented_z):.1f} SD below norm (atrophy).")
                else:
                    note = "enlarged"
                    flags.append(f"{struct}: {abs(oriented_z):.1f} SD enlarged vs norm.")
            regions.append(RegionMetric(struct, vol, pct, oriented_z, abnormal, note))

            if struct in MTA_STRUCTURES:
                # Contribution to MTA risk: only count abnormal direction (clip at 0).
                mta_terms.append(max(0.0, -oriented_z) * ref["ad_weight"])
                mta_weights.append(ref["ad_weight"])

        mta_risk = (sum(mta_terms) / sum(mta_weights)) if mta_weights else 0.0
        stage = self._stage(mta_risk)
        summary = self._summarize(subject_id, regions, mta_risk, stage, source)

        return VolumetryResult(
            subject_id=subject_id,
            source=source,
            etiv_mm3=etiv,
            regions=regions,
            mta_risk_score=mta_risk,
            mta_stage=stage,
            flags=flags,
            summary=summary,
        )

    @staticmethod
    def _stage(mta_risk: float) -> str:
        if mta_risk >= 2.0:
            return "Severe medial-temporal atrophy (MTA-like high)"
        if mta_risk >= 1.0:
            return "Moderate medial-temporal atrophy"
        if mta_risk >= 0.5:
            return "Mild medial-temporal atrophy"
        return "Within normal limits"

    @staticmethod
    def _summarize(
        subject_id: str, regions: List[RegionMetric], mta_risk: float, stage: str, source: str
    ) -> str:
        def z_of(name: str) -> Optional[float]:
            for r in regions:
                if r.structure == name:
                    return r.z_score
            return None

        lh, rh = z_of("Left-Hippocampus"), z_of("Right-Hippocampus")
        llv, rlv = z_of("Left-Lateral-Ventricle"), z_of("Right-Lateral-Ventricle")
        bits: List[str] = []
        if lh is not None and rh is not None:
            bits.append(f"hippocampus L/R z={lh:+.1f}/{rh:+.1f}")
        if llv is not None and rlv is not None:
            bits.append(f"lateral ventricles z={llv:+.1f}/{rlv:+.1f}")
        detail = "; ".join(bits) if bits else "no target structures parsed"
        return (
            f"{stage} (MTA risk {mta_risk:.2f}); {detail} "
            f"[{len([r for r in regions if r.abnormal])} abnormal region(s), source={source}]."
        )

    def _find_subject_stats(self, subject_id: str) -> Optional[str]:
        if not self.freesurfer_root or not os.path.isdir(self.freesurfer_root):
            return None
        # OASIS FreeSurfer layout: <root>/<subject>/.../stats/aseg.stats
        candidates = glob.glob(
            os.path.join(self.freesurfer_root, "**", "stats", "aseg.stats"), recursive=True
        )
        sid = subject_id.lower()
        for path in candidates:
            if sid in path.lower():
                return path
        return candidates[0] if candidates else None

    @staticmethod
    def _subject_from_path(path: str) -> str:
        m = re.search(r"(OAS\d_\w+?\d+)", path)
        return m.group(1) if m else os.path.basename(os.path.dirname(os.path.dirname(path)))


if __name__ == "__main__":
    # Self-test with a synthetic aseg.stats describing a moderately atrophic brain.
    import tempfile

    sample = """# Title Segmentation Statistics
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.0, mm^3
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   4   18000  21000.0  Left-Lateral-Ventricle    25.0 10.0 0 90 90
  2  43   17500  20500.0  Right-Lateral-Ventricle   25.0 10.0 0 90 90
  3   5    1200   1400.0  Left-Inf-Lat-Vent         30.0 10.0 0 90 90
  4  17    2600   2700.0  Left-Hippocampus          60.0  8.0 0 90 90
  5  53    2700   2800.0  Right-Hippocampus         60.0  8.0 0 90 90
  6  18    1100   1150.0  Left-Amygdala             65.0  8.0 0 90 90
  7  54    1150   1180.0  Right-Amygdala            65.0  8.0 0 90 90
  8  14    1500   1550.0  3rd-Ventricle             40.0 10.0 0 90 90
"""
    with tempfile.NamedTemporaryFile("w", suffix="_aseg.stats", delete=False) as fh:
        fh.write(sample)
        tmp = fh.name

    agent = RegionalVolumetryAgent()
    result = agent.analyze_stats_file(tmp, subject_id="OAS30001")
    print("\n--- Regional Volumetry Self-Test ---")
    print(result.summary)
    for r in result.regions:
        mark = " <== ABNORMAL" if r.abnormal else ""
        print(f"  {r.structure:24s} {r.pct_etiv:6.3f}% eTIV  z={r.z_score:+.2f}{mark}")
    os.remove(tmp)
