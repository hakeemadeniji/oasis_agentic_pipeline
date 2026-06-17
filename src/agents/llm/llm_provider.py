"""
Hybrid LLM provider: free local Ollama + cost-tiered Anthropic Claude.

This is the routing layer that makes the pipeline a genuine *agentic* system
that uses ``ANTHROPIC_API_KEY`` for the hard reasoning, while keeping the cheap,
fully-determined work on free local Ollama. The split is deliberate and
cost-driven:

* **free** tasks (grounded summaries, templated narratives, simple structured
  classification) -> local **Ollama** when reachable; these are things a small
  open model does as well as Claude, for $0.
* **cheap** tasks (short structured extraction/labelling) -> **Claude Haiku 4.5**.
* **standard** reasoning (clinical synthesis, cross-modal explanation) ->
  **Claude Sonnet 4.6**.
* **deep** reasoning (differential diagnosis, therapeutic/cure hypothesis
  generation) -> **Claude Opus 4.8** with adaptive thinking.

Every call degrades gracefully: Anthropic -> Ollama -> deterministic caller
fallback, so the system runs (in reduced form) with no API key and no daemon.

Uses the official ``anthropic`` SDK (imported lazily) and ``requests`` for
Ollama. Model IDs and tiers are configurable in :mod:`config`.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from config import get_settings  # noqa: E402

# Task tiers, ordered cheap -> expensive.
TIER_FREE = "free"
TIER_CHEAP = "cheap"
TIER_STANDARD = "standard"
TIER_DEEP = "deep"


@dataclass
class LLMResult:
    text: str
    provider: str          # "anthropic:<model>" | "ollama:<model>" | "template"
    tier: str
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


class HybridLLMProvider:
    """Routes a task to the cheapest capable backend, with graceful fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._anthropic_client = None
        self._anthropic_init_failed = False
        self._edge_ready: Optional[bool] = None

    # ------------------------------------------------------------- backends
    def _get_anthropic(self):
        if self._anthropic_client is not None or self._anthropic_init_failed:
            return self._anthropic_client
        if not self.settings.anthropic_available():
            self._anthropic_init_failed = True
            return None
        try:
            import anthropic  # lazy: optional dependency
            self._anthropic_client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        except Exception as exc:  # pragma: no cover - depends on env
            print(f"[llm_provider] Anthropic unavailable ({exc}); falling back to Ollama/template.")
            self._anthropic_init_failed = True
            self._anthropic_client = None
        return self._anthropic_client

    def _model_for_tier(self, tier: str) -> str:
        return {
            TIER_CHEAP: self.settings.anthropic_model_cheap,
            TIER_STANDARD: self.settings.anthropic_model_standard,
            TIER_DEEP: self.settings.anthropic_model_deep,
        }.get(tier, self.settings.anthropic_model_standard)

    def _ollama_ready(self) -> bool:
        if self._edge_ready is None:
            if not self.settings.enable_llm:
                self._edge_ready = False
            else:
                try:
                    r = requests.get(f"{self.settings.ollama_edge_url.rstrip('/')}/api/tags", timeout=3)
                    self._edge_ready = r.status_code == 200
                except requests.RequestException:
                    self._edge_ready = False
        return self._edge_ready

    # ------------------------------------------------------------- calls
    def _call_anthropic(self, model: str, system: str, prompt: str, tier: str,
                        max_tokens: int) -> Optional[LLMResult]:
        client = self._get_anthropic()
        if client is None:
            return None
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
            if tier == TIER_DEEP:
                # Hard reasoning: let Claude decide thinking depth.
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["output_config"] = {"effort": "high"}
                # Stream long deep outputs to avoid HTTP timeouts.
                with client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                msg = client.messages.create(**kwargs)
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            usage = getattr(msg, "usage", None)
            return LLMResult(
                text=text, provider=f"anthropic:{model}", tier=tier,
                tokens_in=getattr(usage, "input_tokens", 0) or 0,
                tokens_out=getattr(usage, "output_tokens", 0) or 0,
            )
        except Exception as exc:  # pragma: no cover - network/SDK dependent
            print(f"[llm_provider] Anthropic call ({model}) failed: {exc}")
            return None

    def _call_ollama(self, system: str, prompt: str, tier: str) -> Optional[LLMResult]:
        if not self._ollama_ready():
            return None
        url = f"{self.settings.ollama_edge_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.ollama_edge_model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.settings.llm_temperature},
        }
        try:
            r = requests.post(url, json=payload, timeout=self.settings.llm_timeout_seconds)
            r.raise_for_status()
            text = (r.json().get("response") or "").strip()
            if text:
                return LLMResult(text=text, provider=f"ollama:{self.settings.ollama_edge_model}", tier=tier)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            print(f"[llm_provider] Ollama call failed: {exc}")
        return None

    # ------------------------------------------------------------- public
    def complete(
        self,
        system: str,
        prompt: str,
        tier: str = TIER_STANDARD,
        free_capable: bool = False,
        max_tokens: int = 1200,
        template_fallback: Optional[str] = None,
    ) -> LLMResult:
        """
        Route a completion to the cheapest capable backend.

        Args:
            tier: TIER_CHEAP / TIER_STANDARD / TIER_DEEP — the Claude tier to use
                  when Claude handles it.
            free_capable: if True, try free local Ollama first (when
                  ``PREFER_FREE_WHEN_CAPABLE``), since the task doesn't need
                  frontier reasoning.
            template_fallback: deterministic text returned if no backend is up.
        """
        order: List[str] = []
        prefer_free = free_capable and self.settings.prefer_free_when_capable
        if tier == TIER_FREE or prefer_free:
            order = ["ollama", "anthropic"]
        else:
            order = ["anthropic", "ollama"]

        anth_tier = TIER_CHEAP if tier == TIER_FREE else tier
        model = self._model_for_tier(anth_tier)

        for backend in order:
            if backend == "anthropic":
                res = self._call_anthropic(model, system, prompt, anth_tier, max_tokens)
            else:
                res = self._call_ollama(system, prompt, tier)
            if res and res.text:
                return res

        return LLMResult(
            text=template_fallback or "[no language model available]",
            provider="template", tier=tier, error="all_backends_unavailable",
        )

    def status(self) -> Dict[str, Any]:
        return {
            "anthropic": self.settings.anthropic_available(),
            "anthropic_models": {
                "cheap": self.settings.anthropic_model_cheap,
                "standard": self.settings.anthropic_model_standard,
                "deep": self.settings.anthropic_model_deep,
            },
            "ollama": self._ollama_ready(),
            "ollama_model": self.settings.ollama_edge_model,
            "prefer_free_when_capable": self.settings.prefer_free_when_capable,
        }


# Process-wide singleton (cheap to construct; clients are lazy).
_PROVIDER: Optional[HybridLLMProvider] = None


def get_provider() -> HybridLLMProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = HybridLLMProvider()
    return _PROVIDER


if __name__ == "__main__":
    p = get_provider()
    print("Hybrid LLM provider status:")
    print(json.dumps(p.status(), indent=2))
    r = p.complete(
        system="You are a concise assistant.",
        prompt="Say 'ready' in one word.",
        tier=TIER_FREE, free_capable=True,
        template_fallback="ready (template fallback — no LLM backend reachable)",
    )
    print(f"\n[{r.provider}] {r.text}")
