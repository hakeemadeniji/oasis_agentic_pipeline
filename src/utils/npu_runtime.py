"""
ONNX Runtime execution-provider selection for Windows ARM64 / Snapdragon NPU.

This module reconciles the *preferred* provider chain from :mod:`config`
(``QNN -> DirectML -> CPU``) against the providers actually compiled into the
installed ``onnxruntime`` build, and returns a ready-to-use
:class:`onnxruntime.InferenceSession`.

Why this matters
----------------
The same ``.onnx`` artifact can run on:

* the **Snapdragon Hexagon NPU** via ``QNNExecutionProvider`` (lowest power,
  highest throughput for INT8 graphs) -- requires ``onnxruntime-qnn``,
* the **ARM64 GPU** via ``DmlExecutionProvider`` (DirectML),
* or plain **CPU** everywhere else.

Because real edge fleets are heterogeneous, we never assume a provider exists --
we probe ``get_available_providers()`` and degrade gracefully, logging exactly
which silicon ended up serving inference (useful for fleet telemetry).
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional, Tuple

# Allow ``import config`` whether imported from src/ or a sibling package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from config import get_settings  # noqa: E402


def _provider_options(provider: str) -> Dict[str, Any]:
    """Build per-provider session options (QNN/DirectML need extra config)."""
    settings = get_settings()
    if provider == "QNNExecutionProvider":
        # HTP = Hexagon Tensor Processor (the Snapdragon NPU block).
        return {
            "backend_path": settings.qnn_backend_path,
            "htp_performance_mode": settings.qnn_htp_performance_mode,
            "htp_graph_finalization_optimization_mode": "3",
        }
    if provider == "DmlExecutionProvider":
        return {"device_id": 0}
    return {}


def resolve_providers() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Intersect the preferred provider chain with what's actually available.

    Returns a list of ``(provider_name, provider_options)`` tuples in priority
    order, always ending with the CPU provider as a safety net.
    """
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = {"CPUExecutionProvider"}

    preferred = get_settings().onnx_providers
    chain: List[Tuple[str, Dict[str, Any]]] = []
    for provider in preferred:
        if provider in available and provider not in {p for p, _ in chain}:
            chain.append((provider, _provider_options(provider)))

    # Guarantee a CPU fallback so a session can always be created.
    if "CPUExecutionProvider" not in {p for p, _ in chain}:
        chain.append(("CPUExecutionProvider", {}))
    return chain


def describe_runtime() -> str:
    """Human-readable line describing the resolved acceleration path."""
    chain = resolve_providers()
    primary = chain[0][0] if chain else "CPUExecutionProvider"
    label = {
        "QNNExecutionProvider": "Snapdragon NPU (Hexagon/QNN)",
        "DmlExecutionProvider": "ARM64 GPU (DirectML)",
        "CUDAExecutionProvider": "NVIDIA GPU (CUDA)",
        "CPUExecutionProvider": "CPU",
    }.get(primary, primary)
    fallbacks = ", ".join(p for p, _ in chain[1:]) or "none"
    return f"Primary accelerator: {label}  |  Fallback chain: {fallbacks}"


def create_session(model_path: str, sess_options: Optional[Any] = None) -> Any:
    """
    Create an ``InferenceSession`` bound to the best available provider.

    Falls back through the resolved chain automatically; if NPU/GPU providers
    fail to initialize at runtime (e.g. missing driver), ONNX Runtime itself
    rolls back to the next provider in the list.
    """
    import onnxruntime as ort

    if sess_options is None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    chain = resolve_providers()
    providers = [p for p, _ in chain]
    provider_options = [opts for _, opts in chain]

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
    except Exception as exc:  # pragma: no cover - hardware specific
        # Last-resort: pure CPU. Keeps the edge node alive even if the NPU
        # toolchain is misconfigured.
        print(f"[npu_runtime] Provider chain {providers} failed ({exc}). Falling back to CPU.")
        session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    active = session.get_providers()
    print(f"[npu_runtime] {describe_runtime()}")
    print(f"[npu_runtime] Session active providers: {active}")
    return session


if __name__ == "__main__":
    print("Resolved ONNX Runtime providers (preference -> reconciled):")
    for name, opts in resolve_providers():
        print(f"  - {name}  {opts if opts else ''}")
    print(describe_runtime())
