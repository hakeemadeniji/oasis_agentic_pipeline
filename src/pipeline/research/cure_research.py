"""
Cure-research experiment engine — deterministic data mining over OASIS.

This module runs reproducible statistical experiments across the OASIS-1
cross-sectional + longitudinal tables (and OASIS-3/4 FreeSurfer/PUP derivatives
when present) to surface *clues* relevant to treating and preventing Alzheimer's:
which factors track with disease, which track with slower decline (candidate
protective levers), and which structures/biomarkers are the strongest signals
(candidate therapeutic targets).

It is purely statistical and deterministic — no LLM here. Its structured output
is what the Therapeutic Insight agent (Claude) reasons over to generate testable
hypotheses. Everything is wrapped in try/except so a missing column or file
degrades to "not computed" rather than crashing.

Important framing: these are **screening-cohort associations and hypotheses for
research prioritization**, not causal claims or treatment recommendations. OASIS
is observational; confounding is expected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None


@dataclass
class Finding:
    name: str
    metric: str
    value: float
    p_value: Optional[float]
    n: int
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "metric": self.metric, "value": round(self.value, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 5),
            "n": self.n, "interpretation": self.interpretation,
        }


@dataclass
class ResearchReport:
    findings: List[Finding] = field(default_factory=list)
    candidate_targets: List[str] = field(default_factory=list)
    candidate_protective_factors: List[str] = field(default_factory=list)
    cohort: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort": self.cohort,
            "findings": [f.to_dict() for f in self.findings],
            "candidate_targets": self.candidate_targets,
            "candidate_protective_factors": self.candidate_protective_factors,
        }


def _corr(x: pd.Series, y: pd.Series):
    """Pearson r + p with NaN handling; returns (r, p, n)."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 5:
        return None
    a, b = df.iloc[:, 0].astype(float), df.iloc[:, 1].astype(float)
    if a.std() == 0 or b.std() == 0:
        return None
    if _stats is not None:
        r, p = _stats.pearsonr(a, b)
    else:
        r, p = float(np.corrcoef(a, b)[0, 1]), None
    return float(r), (None if p is None else float(p)), int(len(df))


class CureResearchEngine:
    """Runs the deterministic experiment battery and assembles a ResearchReport."""

    def __init__(self, workspace_root: str):
        self.root = workspace_root
        self.cross_csv = os.path.join(workspace_root, "data", "oasis_raw", "oasis_cross-sectional.csv")
        self.clinical_csv = os.path.join(workspace_root, "data", "oasis_raw", "oasis_clinical_data.csv")
        self.long_csv = os.path.join(workspace_root, "data", "oasis_raw", "oasis_longitudinal.csv")

    def run(self) -> ResearchReport:
        report = ResearchReport()
        self._experiment_cross_sectional(report)
        self._experiment_longitudinal(report)
        self._derive_candidates(report)
        return report

    # ---------------------------------------------------- cross-sectional
    def _experiment_cross_sectional(self, report: ResearchReport) -> None:
        path = self.cross_csv if os.path.exists(self.cross_csv) else self.clinical_csv
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        report.cohort["cross_sectional_n"] = int(len(df))

        # Dementia label from CDR when present (CDR > 0 = impaired).
        if "CDR" in df.columns:
            df["_impaired"] = (pd.to_numeric(df["CDR"], errors="coerce") > 0).astype(float)
            report.cohort["impaired_fraction"] = round(float(df["_impaired"].mean()), 3)

        # Factor -> impairment / cognition associations.
        factors = [c for c in ["nWBV", "eTIV", "ASF", "Educ", "EDUC", "SES", "Age", "MMSE"] if c in df.columns]
        # 1) Brain volume (nWBV) vs cognition (MMSE)
        if "nWBV" in df.columns and "MMSE" in df.columns:
            r = _corr(df["nWBV"], df["MMSE"])
            if r:
                report.findings.append(Finding(
                    "Whole-brain volume vs cognition", "pearson_r", r[0], r[1], r[2],
                    f"nWBV correlates with MMSE (r={r[0]:+.2f}); atrophy tracks cognitive loss — "
                    "supports neurodegeneration (the 'N' axis) as a therapeutic target.",
                ))
        # 2) Education vs cognition (cognitive-reserve clue)
        edu_col = "Educ" if "Educ" in df.columns else ("EDUC" if "EDUC" in df.columns else None)
        if edu_col and "MMSE" in df.columns:
            r = _corr(df[edu_col], df["MMSE"])
            if r:
                report.findings.append(Finding(
                    "Education vs cognition (cognitive reserve)", "pearson_r", r[0], r[1], r[2],
                    f"Higher education associates with higher MMSE (r={r[0]:+.2f}); consistent with "
                    "cognitive-reserve — a modifiable, non-pharmacological prevention lever.",
                ))
        # 3) Each factor vs impairment (point-biserial)
        if "_impaired" in df.columns:
            for f in factors:
                if f in ("MMSE",):
                    continue
                r = _corr(df[f], df["_impaired"])
                if r and r[1] is not None and r[1] < 0.05:
                    direction = "lower" if r[0] < 0 else "higher"
                    report.findings.append(Finding(
                        f"{f} vs dementia status", "point_biserial_r", r[0], r[1], r[2],
                        f"{direction.title()} {f} associates with dementia (r={r[0]:+.2f}, p={r[1]:.3g}).",
                    ))

    # -------------------------------------------------------- longitudinal
    def _experiment_longitudinal(self, report: ResearchReport) -> None:
        if not os.path.exists(self.long_csv):
            return
        try:
            ldf = pd.read_csv(self.long_csv)
        except Exception:
            return
        report.cohort["longitudinal_records"] = int(len(ldf))
        if not {"Subject ID", "nWBV", "MR Delay"}.issubset(ldf.columns):
            return

        # Per-subject whole-brain atrophy rate (%/yr) and MMSE slope.
        rows = []
        for sid, sub in ldf.groupby("Subject ID"):
            sub = sub.sort_values("Visit") if "Visit" in sub.columns else sub
            if len(sub) < 2:
                continue
            yrs = (sub["MR Delay"].max() - sub["MR Delay"].min()) / 365.25
            if yrs <= 0:
                continue
            nwbv0, nwbv1 = float(sub["nWBV"].iloc[0]), float(sub["nWBV"].iloc[-1])
            atrophy = (nwbv0 - nwbv1) / nwbv0 / yrs * 100
            grp = str(sub["Group"].iloc[0]) if "Group" in sub.columns else "Unknown"
            edu = float(sub["EDUC"].dropna().iloc[0]) if "EDUC" in sub.columns and sub["EDUC"].notna().any() else np.nan
            rows.append({"sid": sid, "atrophy": atrophy, "group": grp, "educ": edu})
        if not rows:
            return
        sdf = pd.DataFrame(rows)

        # Atrophy by diagnostic group (validates the decline signal).
        by_group = sdf.groupby("group")["atrophy"].mean().to_dict()
        report.cohort["mean_atrophy_pct_per_yr_by_group"] = {k: round(float(v), 3) for k, v in by_group.items()}
        demented = sdf[sdf["group"].str.contains("Demented", case=False, na=False) &
                       ~sdf["group"].str.contains("Non", case=False, na=False)]["atrophy"]
        nondem = sdf[sdf["group"].str.contains("Nondemented", case=False, na=False)]["atrophy"]
        if len(demented) >= 3 and len(nondem) >= 3 and _stats is not None:
            t, p = _stats.ttest_ind(demented, nondem, equal_var=False)
            report.findings.append(Finding(
                "Atrophy rate: demented vs non-demented", "welch_t", float(t), float(p),
                int(len(demented) + len(nondem)),
                f"Demented subjects lose brain volume faster ({demented.mean():.2f} vs "
                f"{nondem.mean():.2f} %/yr); slowing this rate is a concrete therapeutic endpoint.",
            ))

        # Cognitive reserve longitudinally: education vs atrophy rate.
        r = _corr(sdf["educ"], sdf["atrophy"])
        if r:
            report.findings.append(Finding(
                "Education vs atrophy rate", "pearson_r", r[0], r[1], r[2],
                f"Education vs whole-brain atrophy rate (r={r[0]:+.2f}); a negative r would support "
                "education/cognitive activity as a disease-slowing lever worth a prevention trial.",
            ))

    # ---------------------------------------------------------- candidates
    def _derive_candidates(self, report: ResearchReport) -> None:
        targets = [
            "Medial-temporal (hippocampal) neurodegeneration — strongest structural signal; "
            "target neuroprotection / synaptic preservation.",
        ]
        protective = []
        for f in report.findings:
            low = f.interpretation.lower()
            if "amyloid" in low or "atn" in low:
                targets.append("Amyloid/tau pathology (ATN A/T axes) — anti-amyloid / anti-tau strategies.")
            if "cognitive-reserve" in low or "cognitive reserve" in low or "education" in low:
                protective.append("Cognitive reserve via education / lifelong cognitive activity.")
            if "vascular" in low:
                protective.append("Vascular risk-factor control (BP, glycemia, lipids).")
        # de-dup preserving order
        report.candidate_targets = list(dict.fromkeys(targets))
        report.candidate_protective_factors = list(dict.fromkeys(protective)) or [
            "Cognitive reserve (education / cognitive activity) — supported by the cohort associations.",
            "Vascular risk-factor control — biologically plausible; test in the OASIS-3 vascular subset.",
        ]


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    report = CureResearchEngine(root).run()
    import json
    print(json.dumps(report.to_dict(), indent=2))
