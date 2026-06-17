"""
Agent 8: Clinical Reasoner (hybrid edge-cloud LLM narrator).

This agent synthesizes the structured outputs of every other agent (vision,
biomarker, temporal, regional volumetry, RAG, ethicist) into a coherent,
evidence-grounded clinical narrative.

It is deliberately built on **Ollama** so that *all* language reasoning runs on
free, locally-hosted open-weight models -- there are **no metered API keys**
anywhere in the pipeline.

Routing policy (hybrid edge-cloud)
----------------------------------
1. **Edge first** -- a small model on the local Snapdragon device
   (e.g. ``llama3.2:3b`` / ``phi3.5``) handles routine, high-confidence cases.
2. **Cloud escalation** -- if confidence is below the configured floor, or the
   ethicist flagged the case, *and* an optional self-hosted Ollama endpoint is
   configured, the request escalates to a larger model (e.g. ``llama3.1:8b``).
   The "cloud" tier is still self-hosted Ollama -- no paid tokens.
3. **Deterministic fallback** -- if no Ollama daemon is reachable, a templated
   summary is produced so the pipeline never hard-fails on an edge device.

The reasoner is *grounded*: it is only ever asked to summarize and explain the
numbers produced by the deterministic agents. It is explicitly instructed not to
invent a diagnosis -- the authorized classification still comes from the vision
model gated by the ethicist.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from config import get_settings  # noqa: E402


SYSTEM_PROMPT = (
    "You are a board-certified neurology decision-support assistant embedded in "
    "a multi-agent Alzheimer's screening pipeline. You receive the structured, "
    "already-computed outputs of specialized agents (MRI vision model, clinical "
    "biomarkers, longitudinal trajectory, regional brain volumetry, retrieved "
    "guidelines, and an ethics audit). Your job is to write a concise, "
    "evidence-grounded clinical summary for a reviewing clinician.\n\n"
    "Rules:\n"
    "1. NEVER invent or override the diagnosis. The authorized classification is "
    "provided to you; report it faithfully.\n"
    "2. Ground every statement in the numbers you are given. Do not fabricate "
    "values, citations, or biomarkers.\n"
    "3. If the ethics audit flagged the case, lead with that and recommend human "
    "review.\n"
    "4. Be specific about regional volumetry (hippocampus, ventricles, etc.) when "
    "provided.\n"
    "5. Keep it under ~200 words, use plain clinical language, and end with a "
    "one-line recommended next step.\n"
    "This is screening decision-support, not a final diagnosis."
)


@dataclass
class ReasonerResult:
    narrative: str
    tier: str           # "edge" | "cloud" | "template"
    model: str
    escalated: bool


class ClinicalReasonerAgent:
    """Hybrid edge-cloud LLM narrator backed entirely by local Ollama."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = self.settings.enable_llm
        self._edge_ready: Optional[bool] = None
        if self.enabled:
            print(
                f"[*] Clinical Reasoner (Agent 8) -> Ollama edge model "
                f"'{self.settings.ollama_edge_model}' @ {self.settings.ollama_edge_url}"
                + (
                    f" | cloud fallback '{self.settings.ollama_cloud_model}'"
                    if self.settings.cloud_available()
                    else " | cloud fallback disabled"
                )
            )
        else:
            print("[*] Clinical Reasoner (Agent 8) disabled -> deterministic template mode.")

    # ------------------------------------------------------------------ utils
    def _ping(self, base_url: str) -> bool:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def edge_ready(self) -> bool:
        """Cache whether the local Ollama daemon is reachable (avoid re-probing)."""
        if self._edge_ready is None:
            self._edge_ready = self.enabled and self._ping(self.settings.ollama_edge_url)
        return self._edge_ready

    def _call_ollama(self, base_url: str, model: str, prompt: str) -> Optional[str]:
        """Single non-streaming Ollama generate call. Returns None on failure."""
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.settings.llm_temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.settings.llm_timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            return text or None
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            print(f"[llm_reasoner] Ollama call to {model}@{base_url} failed: {exc}")
            return None

    # ----------------------------------------------------------------- routing
    def _should_escalate(self, evidence: Dict[str, Any]) -> bool:
        if not self.settings.cloud_available():
            return False
        if self.settings.prefer_cloud:
            return True
        confidence = float(evidence.get("confidence", 100.0))
        flagged = bool(evidence.get("ethics_flagged", False))
        return flagged or confidence < self.settings.cloud_escalation_confidence

    # --------------------------------------------------------------- narrative
    def synthesize(self, evidence: Dict[str, Any]) -> ReasonerResult:
        """
        Produce a clinical narrative from the consolidated agent evidence dict.

        ``evidence`` is a flat dict of the structured outputs (prediction,
        confidence, mmse, age, atrophy velocity, regional volumetry summary,
        rag context, ethics verdict).
        """
        prompt = self._build_prompt(evidence)

        if not self.enabled:
            return ReasonerResult(self._template(evidence), "template", "deterministic", False)

        escalate = self._should_escalate(evidence)

        # Cloud tier (still self-hosted Ollama) when escalation is warranted.
        if escalate and self.settings.cloud_available():
            text = self._call_ollama(
                self.settings.ollama_cloud_url, self.settings.ollama_cloud_model, prompt
            )
            if text:
                return ReasonerResult(text, "cloud", self.settings.ollama_cloud_model, True)

        # Edge tier (local Snapdragon device).
        if self.edge_ready():
            text = self._call_ollama(
                self.settings.ollama_edge_url, self.settings.ollama_edge_model, prompt
            )
            if text:
                return ReasonerResult(text, "edge", self.settings.ollama_edge_model, False)

        # Last resort cloud try if edge is down but escalation wasn't triggered.
        if self.settings.cloud_available():
            text = self._call_ollama(
                self.settings.ollama_cloud_url, self.settings.ollama_cloud_model, prompt
            )
            if text:
                return ReasonerResult(text, "cloud", self.settings.ollama_cloud_model, True)

        # Deterministic fallback keeps the edge node operational offline.
        return ReasonerResult(self._template(evidence), "template", "deterministic", False)

    def _build_prompt(self, e: Dict[str, Any]) -> str:
        rag_context = e.get("rag_context") or []
        if isinstance(rag_context, str):
            rag_context = [rag_context]
        lines = [
            "Synthesize a clinical screening summary from these agent outputs:",
            "",
            f"- Authorized classification: {e.get('authorized_class', e.get('prediction', 'N/A'))}",
            f"- Vision model raw prediction: {e.get('prediction', 'N/A')} "
            f"({float(e.get('confidence', 0.0)):.1f}% confidence)",
            f"- Age: {e.get('age', 'N/A')} | MMSE: {e.get('mmse', 'N/A')}/30",
            f"- Longitudinal trend: {e.get('clinical_trend', 'N/A')} "
            f"(atrophy velocity {float(e.get('atrophy_velocity', 0.0)):.2f}%/yr)",
            f"- Regional volumetry: {e.get('volumetry_summary', 'not available')}",
            f"- Ethics audit: {'FLAGGED - ' if e.get('ethics_flagged') else 'cleared - '}"
            f"{e.get('ethics_message', 'n/a')}",
            "- Retrieved guideline context:",
        ]
        for ctx in rag_context[:3]:
            lines.append(f"    * {ctx}")
        lines.append("")
        lines.append("Write the summary now.")
        return "\n".join(lines)

    def _template(self, e: Dict[str, Any]) -> str:
        """Deterministic, dependency-free narrative used when no LLM is reachable."""
        flagged = bool(e.get("ethics_flagged"))
        head = (
            "[HUMAN REVIEW REQUIRED] " if flagged else ""
        )
        cls = e.get("authorized_class", e.get("prediction", "Undetermined"))
        conf = float(e.get("confidence", 0.0))
        mmse = e.get("mmse", "N/A")
        vol = e.get("volumetry_summary", "regional volumetry not available")
        trend = e.get("clinical_trend", "N/A")
        atrophy = float(e.get("atrophy_velocity", 0.0))
        narrative = (
            f"{head}Screening summary: imaging-based classification is '{cls}' at "
            f"{conf:.1f}% model confidence, with an MMSE of {mmse}/30. Longitudinal "
            f"assessment shows '{trend}' (whole-brain atrophy ~{atrophy:.2f}%/yr). "
            f"Regional volumetry: {vol}. "
        )
        if flagged:
            narrative += (
                f"The ethics guardrail flagged this case ({e.get('ethics_message', '')}); "
                "automated output is withheld pending neurologist review. "
                "Recommended next step: route to human expert for adjudication."
            )
        else:
            narrative += (
                "Cross-modal signals are consistent. Recommended next step: "
                "clinician confirmation and routine follow-up imaging per guideline cadence."
            )
        return narrative


if __name__ == "__main__":
    agent = ClinicalReasonerAgent()
    demo = {
        "prediction": "Very mild Dementia",
        "authorized_class": "Very mild Dementia",
        "confidence": 78.4,
        "age": 74.2,
        "mmse": 25.0,
        "clinical_trend": "Typical Age-Related Neuro-Degradation",
        "atrophy_velocity": 0.62,
        "volumetry_summary": "L/R hippocampus -1.8 SD below normative mean; lateral ventricles +1.4 SD enlarged.",
        "ethics_flagged": False,
        "ethics_message": "VERIFIED: data streams aligned.",
        "rag_context": ["Very Mild Dementia (CDR 0.5) presents with slight memory complaints and MMSE typically above 25."],
    }
    result = agent.synthesize(demo)
    print(f"\n--- Clinical Reasoner Output [tier={result.tier}, model={result.model}] ---\n")
    print(result.narrative)
