"""
Agent 11: Differential Diagnosis Reasoner (Claude, deep tier).

A genuinely complex clinical-reasoning task that benefits from a frontier model:
given the consolidated multi-modal evidence (MRI severity class + confidence,
MMSE, age, longitudinal atrophy, regional volumetry, ATN biomarker profile),
produce a *ranked differential* across the major dementia etiologies, each with a
calibrated likelihood, a one-line rationale grounded in the supplied numbers, and
the confirmatory work-up that would resolve it.

This runs on the **deep** Claude tier (Opus) through the hybrid provider, because
weighing competing etiologies against partial biomarker evidence is exactly the
kind of nuanced reasoning small local models do poorly. It still degrades: a
deterministic rule-based ranking is always computed and is used both as a
grounding prior in the prompt and as the offline fallback.

Output is parsed JSON; if the model returns unparseable text the deterministic
ranking is returned so the pipeline never breaks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from agents.llm.llm_provider import get_provider, TIER_DEEP  # noqa: E402

ETIOLOGIES = [
    "Alzheimer's disease",
    "Frontotemporal dementia",
    "Dementia with Lewy bodies",
    "Vascular cognitive impairment",
    "Normal aging / non-neurodegenerative",
]

SYSTEM_PROMPT = (
    "You are a neurology differential-diagnosis assistant in a screening pipeline. "
    "You are given the already-computed outputs of imaging, biomarker, volumetry, "
    "and ATN agents. Produce a ranked differential across these etiologies: "
    + "; ".join(ETIOLOGIES)
    + ". "
    "Rules: (1) ground every likelihood in the supplied numbers — do not invent "
    "values; (2) likelihoods are 0-100 and should sum to ~100; (3) give a one-line "
    "rationale per etiology citing the specific evidence; (4) list the confirmatory "
    "work-up that would most change the ranking; (5) this is screening decision "
    "support, not a diagnosis. Respond with ONLY a JSON object of the form: "
    '{"ranking": [{"etiology": str, "likelihood": int, "rationale": str}], '
    '"recommended_workup": [str], "summary": str}.'
)


@dataclass
class DifferentialResult:
    ranking: List[Dict[str, Any]]
    recommended_workup: List[str]
    summary: str
    provider: str
    grounded_prior: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ranking": self.ranking,
            "recommended_workup": self.recommended_workup,
            "summary": self.summary,
            "provider": self.provider,
        }


class DifferentialDiagnosisAgent:
    """Claude-powered ranked differential with a deterministic rule-based prior."""

    def __init__(self) -> None:
        self.provider = get_provider()

    # --------------------------------------------------- deterministic prior
    @staticmethod
    def _prior(e: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rule-based likelihood prior from the structured evidence (offline-safe)."""
        cls = str(e.get("prediction", "")).lower()
        mmse = float(e.get("mmse", 30) or 30)
        atrophy = float(e.get("atrophy_velocity", 0) or 0)
        a = str(e.get("atn_a", "indeterminate"))
        t = str(e.get("atn_t", "indeterminate"))
        n = str(e.get("atn_n", "indeterminate"))
        hippo_z = e.get("hippocampus_z")

        scores = {k: 10.0 for k in ETIOLOGIES}
        # Alzheimer's: amyloid+ and/or tau+ and medial-temporal atrophy
        if a == "positive":
            scores["Alzheimer's disease"] += 35
        if t == "positive":
            scores["Alzheimer's disease"] += 25
        if hippo_z is not None and float(hippo_z) <= -1.5:
            scores["Alzheimer's disease"] += 20
        if "dementia" in cls and "non" not in cls:
            scores["Alzheimer's disease"] += 10
        # Vascular: stepwise/atrophy with amyloid-
        if a == "negative" and n == "positive":
            scores["Vascular cognitive impairment"] += 20
            scores["Frontotemporal dementia"] += 12
            scores["Dementia with Lewy bodies"] += 10
        # Normal aging: intact cognition, no neurodegeneration
        if mmse >= 27 and n != "positive":
            scores["Normal aging / non-neurodegenerative"] += 35
        if atrophy < 0.5 and n != "positive":
            scores["Normal aging / non-neurodegenerative"] += 10
        # FTD hint: marked atrophy, younger, relatively preserved memory
        if n == "positive" and (a != "positive"):
            scores["Frontotemporal dementia"] += 8

        total = sum(scores.values()) or 1.0
        ranked = sorted(
            ({"etiology": k, "likelihood": round(v / total * 100)} for k, v in scores.items()),
            key=lambda d: d["likelihood"],
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------- analyze
    def analyze(self, evidence: Dict[str, Any]) -> DifferentialResult:
        prior = self._prior(evidence)
        prompt = self._build_prompt(evidence, prior)
        fallback = self._fallback(evidence, prior)

        result = self.provider.complete(
            system=SYSTEM_PROMPT,
            prompt=prompt,
            tier=TIER_DEEP,  # complex reasoning -> Opus when available
            free_capable=False,
            max_tokens=1400,
            template_fallback="",  # we build our own structured fallback below
        )
        parsed = self._parse(result.text)
        if parsed is None:
            return DifferentialResult(
                ranking=fallback["ranking"],
                recommended_workup=fallback["recommended_workup"],
                summary=fallback["summary"],
                provider=result.provider if result.text else "template",
                grounded_prior=prior,
            )
        return DifferentialResult(
            ranking=parsed.get("ranking", fallback["ranking"]),
            recommended_workup=parsed.get("recommended_workup", fallback["recommended_workup"]),
            summary=parsed.get("summary", fallback["summary"]),
            provider=result.provider,
            grounded_prior=prior,
        )

    # ------------------------------------------------------------- helpers
    def _build_prompt(self, e: Dict[str, Any], prior: List[Dict[str, Any]]) -> str:
        lines = [
            "Patient evidence (already computed by upstream agents):",
            f"- MRI vision class: {e.get('prediction', 'N/A')} ({float(e.get('confidence', 0)):.0f}% conf)",
            f"- Age: {e.get('age', 'N/A')} | MMSE: {e.get('mmse', 'N/A')}/30",
            f"- Whole-brain atrophy velocity: {float(e.get('atrophy_velocity', 0)):.2f} %/yr",
            f"- Hippocampus z-score: {e.get('hippocampus_z', 'n/a')}",
            f"- Regional volumetry: {e.get('volumetry_summary', 'n/a')}",
            f"- ATN profile: A={e.get('atn_a', '?')} T={e.get('atn_t', '?')} N={e.get('atn_n', '?')} "
            f"({e.get('atn_category', 'n/a')})",
            "",
            "A deterministic rule-based prior (for grounding, refine as warranted):",
            json.dumps(prior),
            "",
            "Return the JSON object now.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse(text: str):
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "ranking" in obj:
                return obj
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _fallback(e: Dict[str, Any], prior: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = prior[0]["etiology"] if prior else "Undetermined"
        workup = []
        if e.get("atn_a", "indeterminate") != "positive":
            workup.append("Amyloid PET or CSF Aβ42/40 to establish the A axis")
        if e.get("atn_t", "indeterminate") != "positive":
            workup.append("Tau PET (AV1451) or CSF p-tau to establish the T axis")
        workup.append("FDG-PET for hypometabolism pattern (AD vs FTD vs DLB)")
        workup.append("Formal neuropsychological testing and clinical history")
        rationale = {
            d["etiology"]: f"prior likelihood {d['likelihood']}% from biomarker/atrophy rules"
            for d in prior
        }
        ranking = [
            {
                "etiology": d["etiology"],
                "likelihood": d["likelihood"],
                "rationale": rationale[d["etiology"]],
            }
            for d in prior
        ]
        return {
            "ranking": ranking,
            "recommended_workup": workup,
            "summary": (
                f"Leading consideration: {top}. Deterministic rule-based ranking "
                "(no LLM backend reachable). Confirm with the recommended work-up."
            ),
        }


if __name__ == "__main__":
    agent = DifferentialDiagnosisAgent()
    demo = {
        "prediction": "Mild Dementia",
        "confidence": 88.0,
        "age": 77,
        "mmse": 21,
        "atrophy_velocity": 1.1,
        "hippocampus_z": -2.2,
        "volumetry_summary": "Moderate medial-temporal atrophy; ventricles enlarged.",
        "atn_a": "positive",
        "atn_t": "positive",
        "atn_n": "positive",
        "atn_category": "Alzheimer's disease (A+T+N+)",
    }
    res = agent.analyze(demo)
    print(f"\n--- Differential [{res.provider}] ---")
    for r in res.ranking:
        print(f"  {r['likelihood']:>3}%  {r['etiology']}")
    print("Workup:", "; ".join(res.recommended_workup))
    print("Summary:", res.summary)
