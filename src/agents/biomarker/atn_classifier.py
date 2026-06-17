"""
Agent 10: ATN Biomarker Profiler (NIA-AA 2018 research framework).

Classifies a subject along the three orthogonal biomarker axes used in modern
Alzheimer's research staging:

    A  — Amyloid       (amyloid PET; SUVR -> Centiloid)
    T  — Tau           (tau PET, e.g. AV1451, meta-ROI SUVR)
    N  — Neurodegeneration (structural atrophy: hippocampal z / nWBV / MTA risk)

and maps the A/T/N profile to a NIA-AA biological category. This is the
single most clinically meaningful upgrade unlocked by the OASIS-3 / OASIS-4 PUP
(PET Unified Pipeline) + FreeSurfer derivatives.

It is designed to **degrade gracefully**: when PET is unavailable (e.g. OASIS-1),
A and T are reported as "indeterminate" and the profile is driven by the N axis
from the regional-volumetry agent, with that limitation stated explicitly.

Thresholds and Centiloid coefficients are configurable screening priors drawn
from the literature; they are calibration/pipeline dependent and must be
site-validated before any clinical use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Tracer-specific linear SUVR -> Centiloid transforms (slope, intercept).
# Centiloid anchors normal=0, typical AD=100. Coefficients are approximate,
# pipeline-dependent priors (PIB per Klunk 2015; florbetapir per Navitsky 2018).
CENTILOID_COEFFS: Dict[str, Dict[str, float]] = {
    "PIB":   {"slope": 93.7, "intercept": -94.6},
    "AV45":  {"slope": 196.9, "intercept": -196.0},   # florbetapir
    "FBP":   {"slope": 196.9, "intercept": -196.0},
    "AV1451": {"slope": 0.0, "intercept": 0.0},        # tau tracer, not amyloid
}

# Default positivity thresholds (screening priors).
AMYLOID_POSITIVE_CENTILOID = 20.0   # ~A+ at >= 20 CL
TAU_POSITIVE_SUVR = 1.27            # entorhinal/temporal meta-ROI AV1451
N_POSITIVE_HIPPO_Z = -1.5           # hippocampal volume z
N_POSITIVE_MTA = 1.0                # medial-temporal-atrophy composite


@dataclass
class ATNResult:
    a_status: str            # positive | negative | indeterminate
    t_status: str
    n_status: str
    centiloid: Optional[float]
    profile: str             # e.g. "A+T-N+"
    category: str            # NIA-AA biological category
    on_ad_continuum: Optional[bool]
    summary: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "a_status": self.a_status,
            "t_status": self.t_status,
            "n_status": self.n_status,
            "centiloid": None if self.centiloid is None else round(self.centiloid, 1),
            "profile": self.profile,
            "category": self.category,
            "on_ad_continuum": self.on_ad_continuum,
            "summary": self.summary,
        }


class ATNBiomarkerProfiler:
    """Maps amyloid / tau / neurodegeneration evidence to NIA-AA categories."""

    def __init__(
        self,
        amyloid_threshold_cl: float = AMYLOID_POSITIVE_CENTILOID,
        tau_threshold_suvr: float = TAU_POSITIVE_SUVR,
        n_hippo_z: float = N_POSITIVE_HIPPO_Z,
        n_mta: float = N_POSITIVE_MTA,
    ) -> None:
        self.amyloid_threshold_cl = amyloid_threshold_cl
        self.tau_threshold_suvr = tau_threshold_suvr
        self.n_hippo_z = n_hippo_z
        self.n_mta = n_mta

    # ------------------------------------------------------------- centiloid
    @staticmethod
    def suvr_to_centiloid(suvr: float, tracer: str = "PIB") -> float:
        c = CENTILOID_COEFFS.get(tracer.upper(), CENTILOID_COEFFS["PIB"])
        return c["slope"] * suvr + c["intercept"]

    # ------------------------------------------------------------- classify
    def classify(
        self,
        amyloid_suvr: Optional[float] = None,
        amyloid_centiloid: Optional[float] = None,
        amyloid_tracer: str = "PIB",
        tau_suvr: Optional[float] = None,
        hippocampus_z: Optional[float] = None,
        mta_risk: Optional[float] = None,
        nwbv: Optional[float] = None,
    ) -> ATNResult:
        # --- A axis ---
        centiloid = amyloid_centiloid
        if centiloid is None and amyloid_suvr is not None:
            centiloid = self.suvr_to_centiloid(amyloid_suvr, amyloid_tracer)
        if centiloid is None:
            a_status = "indeterminate"
        else:
            a_status = "positive" if centiloid >= self.amyloid_threshold_cl else "negative"

        # --- T axis ---
        if tau_suvr is None:
            t_status = "indeterminate"
        else:
            t_status = "positive" if tau_suvr >= self.tau_threshold_suvr else "negative"

        # --- N axis (atrophy; any available structural signal) ---
        n_signals = []
        if hippocampus_z is not None:
            n_signals.append(hippocampus_z <= self.n_hippo_z)
        if mta_risk is not None:
            n_signals.append(mta_risk >= self.n_mta)
        if nwbv is not None:
            n_signals.append(((nwbv - 0.73) / 0.04) <= self.n_hippo_z)
        if not n_signals:
            n_status = "indeterminate"
        else:
            n_status = "positive" if any(n_signals) else "negative"

        profile = f"A{self._sym(a_status)}T{self._sym(t_status)}N{self._sym(n_status)}"
        category, continuum = self._category(a_status, t_status, n_status)
        summary = self._summary(a_status, t_status, n_status, centiloid, category)
        return ATNResult(a_status, t_status, n_status, centiloid, profile, category, continuum, summary)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _sym(status: str) -> str:
        return {"positive": "+", "negative": "-", "indeterminate": "?"}[status]

    def _category(self, a: str, t: str, n: str):
        """Map A/T/N to a NIA-AA biological category and AD-continuum flag."""
        if a == "indeterminate":
            # No amyloid measure: fall back to a neurodegeneration-only descriptor.
            if n == "positive":
                return ("Neurodegeneration present; amyloid status unknown (PET needed to stage)", None)
            if n == "negative":
                return ("No neurodegeneration; amyloid/tau status unknown (PET needed)", None)
            return ("Biomarker profile indeterminate (insufficient data)", None)

        if a == "negative":
            if t == "positive" or n == "positive":
                # A-, but T+ and/or N+
                if t in ("negative", "indeterminate") and n == "positive":
                    return ("Suspected non-Alzheimer's pathophysiology (SNAP)", False)
                return ("Non-AD pathologic change", False)
            return ("Normal AD biomarkers", False)

        # a == positive  -> on the Alzheimer's continuum
        if t == "positive":
            if n == "positive":
                return ("Alzheimer's disease (A+T+N+)", True)
            return ("Alzheimer's disease (A+T+)", True)
        if t == "negative":
            if n == "positive":
                return ("Alzheimer's pathologic change + concomitant neurodegeneration (A+T-N+)", True)
            return ("Alzheimer's pathologic change (A+T-)", True)
        # tau indeterminate
        return ("Alzheimer's continuum — amyloid positive; tau PET needed to stage", True)

    @staticmethod
    def _summary(a: str, t: str, n: str, centiloid: Optional[float], category: str) -> str:
        cl = f"{centiloid:.0f} CL" if centiloid is not None else "amyloid n/a"
        return (
            f"ATN profile: A={a}, T={t}, N={n} ({cl}). Biological category: {category}. "
            "Screening biomarker staging — confirm with a clinician."
        )


if __name__ == "__main__":
    profiler = ATNBiomarkerProfiler()
    print("--- ATN Biomarker Profiler self-test ---\n")

    # 1) Full PET + MRI: amyloid+ (high SUVR), tau+, atrophy+  -> Alzheimer's disease
    r = profiler.classify(amyloid_suvr=1.6, amyloid_tracer="PIB", tau_suvr=1.45, hippocampus_z=-2.1)
    print(r.profile, "->", r.category, f"({r.centiloid:.0f} CL)")

    # 2) MRI only (OASIS-1 style): atrophy+, no PET -> N-driven, A/T unknown
    r = profiler.classify(hippocampus_z=-1.9, mta_risk=1.6)
    print(r.profile, "->", r.category)

    # 3) Amyloid+ only, tau/N normal -> Alzheimer's pathologic change
    r = profiler.classify(amyloid_centiloid=42.0, tau_suvr=1.1, hippocampus_z=-0.4)
    print(r.profile, "->", r.category)

    # 4) Amyloid-, atrophy+ -> SNAP
    r = profiler.classify(amyloid_centiloid=8.0, tau_suvr=1.0, hippocampus_z=-2.0)
    print(r.profile, "->", r.category)
