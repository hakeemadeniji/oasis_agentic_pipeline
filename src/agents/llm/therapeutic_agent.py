"""
Agent 12: Therapeutic Insight / Cure-Research Reasoner (Claude, deep tier).

Reasons over the deterministic findings from
:mod:`pipeline.research.cure_research` to generate **testable research
hypotheses** about treating, slowing, or preventing Alzheimer's — each grounded
in a specific OASIS finding, with a mechanism and a concrete next experiment
(in-silico on OASIS-3/4, or a wet-lab / trial design).

This is the most open-ended reasoning task in the pipeline, so it runs on the
**deep** Claude tier (Opus) through the hybrid provider; for cost control it can
be pointed at a cheaper tier or run in batch. It degrades to a deterministic
hypothesis list built directly from the research report when no LLM is reachable.

Hard framing (enforced in the system prompt and the fallback): these are
hypotheses for **research prioritization** on observational data, explicitly NOT
clinical, treatment, or diagnostic advice. Causation is not claimed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from agents.llm.llm_provider import get_provider, TIER_DEEP  # noqa: E402

SYSTEM_PROMPT = (
    "You are a translational-neuroscience research assistant. You are given "
    "deterministic statistical findings mined from the OASIS observational "
    "cohorts (structural MRI, longitudinal atrophy, cognition, and — when "
    "present — amyloid/tau PET). Propose testable research hypotheses for "
    "treating, slowing, or preventing Alzheimer's disease. "
    "STRICT RULES: (1) ground each hypothesis in a SPECIFIC finding provided — "
    "cite it; (2) OASIS is observational — state hypotheses as associations to "
    "test, never causal or treatment claims; (3) for each, give the proposed "
    "mechanism and a CONCRETE next experiment (an in-silico analysis on "
    "OASIS-3/4, or a wet-lab / clinical-trial design); (4) rank by strength of "
    "the supporting evidence and tractability; (5) this is research "
    "prioritization, NOT medical advice. "
    'Respond with ONLY JSON: {"hypotheses": [{"title": str, "evidence": str, '
    '"mechanism": str, "next_experiment": str, "strength": "high|medium|low"}], '
    '"summary": str}.'
)


@dataclass
class TherapeuticResult:
    hypotheses: List[Dict[str, Any]]
    summary: str
    provider: str

    def to_dict(self) -> Dict[str, Any]:
        return {"hypotheses": self.hypotheses, "summary": self.summary, "provider": self.provider}


class TherapeuticInsightAgent:
    """Claude-powered hypothesis generation grounded on the cure-research report."""

    def __init__(self) -> None:
        self.provider = get_provider()

    def analyze(self, research_report: Dict[str, Any]) -> TherapeuticResult:
        prompt = self._build_prompt(research_report)
        fallback = self._fallback(research_report)
        result = self.provider.complete(
            system=SYSTEM_PROMPT,
            prompt=prompt,
            tier=TIER_DEEP,  # open-ended hypothesis generation -> Opus
            free_capable=False,
            max_tokens=2000,
            template_fallback="",
        )
        parsed = self._parse(result.text)
        if parsed is None:
            return TherapeuticResult(
                fallback["hypotheses"],
                fallback["summary"],
                result.provider if result.text else "template",
            )
        return TherapeuticResult(
            hypotheses=parsed.get("hypotheses", fallback["hypotheses"]),
            summary=parsed.get("summary", fallback["summary"]),
            provider=result.provider,
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _build_prompt(report: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "Cohort summary:",
                json.dumps(report.get("cohort", {}), indent=2),
                "",
                "Deterministic findings:",
                json.dumps(report.get("findings", []), indent=2),
                "",
                "Candidate targets (from the mining):",
                json.dumps(report.get("candidate_targets", []), indent=2),
                "Candidate protective factors:",
                json.dumps(report.get("candidate_protective_factors", []), indent=2),
                "",
                "Generate the ranked, evidence-grounded research hypotheses now (JSON only).",
            ]
        )

    @staticmethod
    def _parse(text: str):
        if not text:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) and "hypotheses" in obj else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback(report: Dict[str, Any]) -> Dict[str, Any]:
        findings = report.get("findings", [])
        hyps: Dict[str, Dict[str, Any]] = {}  # keyed by title to de-duplicate
        for f in findings:
            name = str(f.get("name", ""))
            interp = str(f.get("interpretation", ""))
            blob = (name + " " + interp).lower()
            evidence = f"{name}: {interp}"
            if (
                ("atrophy rate" in blob)
                or ("brain volume faster" in blob)
                or ("therapeutic endpoint" in blob)
            ):
                hyps.setdefault(
                    "Slow whole-brain atrophy rate as a disease-modification endpoint",
                    {
                        "title": "Slow whole-brain atrophy rate as a disease-modification endpoint",
                        "evidence": evidence,
                        "mechanism": "Neurodegeneration (synaptic/neuronal loss) drives volume loss; "
                        "neuroprotective or anti-amyloid/tau agents may slow it.",
                        "next_experiment": "In-silico: model nWBV trajectories on OASIS-3 by ATN status to "
                        "estimate the atrophy-slowing effect size a trial would need.",
                        "strength": "high",
                    },
                )
            if ("cognitive reserve" in blob) or ("education" in blob):
                hyps.setdefault(
                    "Cognitive reserve as a modifiable prevention lever",
                    {
                        "title": "Cognitive reserve as a modifiable prevention lever",
                        "evidence": evidence,
                        "mechanism": "Higher reserve buffers cognition at a given level of pathology, "
                        "delaying symptom onset.",
                        "next_experiment": "Stratify OASIS-3 by education and test whether higher education "
                        "predicts preserved MMSE at matched hippocampal-atrophy levels; "
                        "if so, design a cognitive-activity prevention trial.",
                        "strength": "medium",
                    },
                )
            if ("amyloid" in blob) or ("tau" in blob) or ("atn" in blob):
                hyps.setdefault(
                    "Target amyloid/tau pathology (ATN A/T axes)",
                    {
                        "title": "Target amyloid/tau pathology (ATN A/T axes)",
                        "evidence": evidence,
                        "mechanism": "Amyloid and tau accumulation are upstream drivers; clearing them may "
                        "slow downstream neurodegeneration.",
                        "next_experiment": "On OASIS-3 PUP, test whether amyloid/tau SUVR predicts subsequent "
                        "atrophy/cognition decline to size anti-amyloid/anti-tau effects.",
                        "strength": "medium",
                    },
                )
        if not hyps:
            hyps["Medial-temporal neuroprotection"] = {
                "title": "Medial-temporal neuroprotection",
                "evidence": "Hippocampal atrophy is the strongest structural signal in the cohort.",
                "mechanism": "Preserving hippocampal neurons/synapses should preserve memory.",
                "next_experiment": "Correlate hippocampal-volume trajectories with cognition on OASIS-3 "
                "FreeSurfer to prioritize neuroprotective targets.",
                "strength": "medium",
            }
        order = {"high": 0, "medium": 1, "low": 2}
        ranked = sorted(hyps.values(), key=lambda h: order.get(h["strength"], 3))
        return {
            "hypotheses": ranked,
            "summary": (
                "Deterministic hypothesis set derived from the OASIS associations "
                "(no LLM backend reachable). Research prioritization only — observational "
                "data, not causal or treatment claims."
            ),
        }


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.append(os.path.join(root, "src"))
    from pipeline.research.cure_research import CureResearchEngine

    report = CureResearchEngine(root).run().to_dict()
    agent = TherapeuticInsightAgent()
    res = agent.analyze(report)
    print(f"\n--- Therapeutic Insight [{res.provider}] ---")
    for h in res.hypotheses:
        print(f"  [{h.get('strength', '?'):>6}] {h.get('title')}")
        print(f"           next: {h.get('next_experiment')}")
    print("Summary:", res.summary)
