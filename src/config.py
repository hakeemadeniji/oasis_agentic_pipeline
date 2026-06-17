"""
Central configuration for the OASIS Agentic Pipeline.

This module turns the pipeline into a *hybrid edge-cloud* system that is tuned
for **Windows on ARM64 / Snapdragon X NPU** hardware while keeping every agent
on free, locally-hosted inference (Ollama) instead of metered cloud API keys.

Everything is driven by environment variables (see ``.env.example``) so the same
code runs unchanged on a developer laptop, a Snapdragon edge device, or a
self-hosted "cloud" Ollama box.

Design goals
------------
* **No paid API tokens.** Language reasoning is served by a local Ollama daemon
  (edge) with an optional self-hosted Ollama endpoint as the "cloud" tier.
* **NPU first.** ONNX Runtime execution providers are ordered
  ``QNN (Snapdragon NPU) -> DirectML (ARM64 GPU) -> CPU`` with automatic
  fallback, so the exact same artifact accelerates on hardware that has it and
  still runs everywhere else.
* **Graceful degradation.** If Ollama is not installed/running, the pipeline
  falls back to a deterministic template narrator and never hard-fails.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_arm64() -> bool:
    """True when running on an ARM64 machine (Snapdragon / Windows on ARM, Apple, etc.)."""
    machine = platform.machine().lower()
    return machine in {"arm64", "aarch64"}


def default_onnx_providers() -> List[str]:
    """
    Build the preferred ONNX Runtime execution-provider chain.

    On Windows ARM64 we want the Qualcomm NPU (QNNExecutionProvider) first, then
    the DirectML GPU path, then CPU. The actual availability is reconciled at
    runtime against ``onnxruntime.get_available_providers`` in
    :mod:`utils.npu_runtime`, so listing a provider here is only a *preference*.
    """
    if platform.system() == "Windows":
        if is_arm64():
            return ["QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        return ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    # Linux / macOS edge boxes
    return ["QNNExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@dataclass
class EdgeCloudSettings:
    """Resolved runtime settings for the hybrid edge-cloud architecture."""

    # ---- Hardware / inference acceleration --------------------------------
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "auto"))
    onnx_providers: List[str] = field(
        default_factory=lambda: _env_list("ONNX_PROVIDERS", default_onnx_providers())
    )
    # QNN backend .dll used by the Snapdragon NPU (HTP = Hexagon Tensor Processor).
    qnn_backend_path: str = field(
        default_factory=lambda: os.environ.get("QNN_BACKEND_PATH", "QnnHtp.dll")
    )
    qnn_htp_performance_mode: str = field(
        default_factory=lambda: os.environ.get("QNN_HTP_PERFORMANCE_MODE", "high_performance")
    )

    # ---- Anthropic API (cost-tiered; used for the complex agentic tasks) --
    # The pipeline is HYBRID: free local Ollama handles tasks it does as well,
    # and Claude handles the hard reasoning. Tiers are chosen for cost-control
    # (you decide the spend): cheap=Haiku, standard=Sonnet, deep=Opus.
    enable_anthropic: bool = field(default_factory=lambda: _env_bool("ENABLE_ANTHROPIC", True))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model_cheap: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL_CHEAP", "claude-haiku-4-5")
    )
    anthropic_model_standard: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL_STANDARD", "claude-sonnet-4-6")
    )
    anthropic_model_deep: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_MODEL_DEEP", "claude-opus-4-8")
    )
    # When True, tasks flagged "free-capable" prefer local Ollama before paying
    # for Claude — the cost-effective default.
    prefer_free_when_capable: bool = field(
        default_factory=lambda: _env_bool("PREFER_FREE_WHEN_CAPABLE", True)
    )

    # ---- Language reasoning (Ollama, free local tier) ---------------------
    enable_llm: bool = field(default_factory=lambda: _env_bool("ENABLE_LLM", True))
    # Edge tier: local Ollama daemon on the Snapdragon device.
    ollama_edge_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_EDGE_URL", "http://localhost:11434")
    )
    ollama_edge_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_EDGE_MODEL", "llama3.2:3b")
    )
    # Cloud tier: an *optional* self-hosted Ollama endpoint (still free / no API key).
    ollama_cloud_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_CLOUD_URL", ""))
    ollama_cloud_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_CLOUD_MODEL", "llama3.1:8b")
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0)
    )
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.2))

    # ---- Edge -> cloud escalation policy ----------------------------------
    # When the edge prediction confidence drops below this floor (or an agent is
    # flagged by the ethicist), the reasoner escalates to the cloud tier if one
    # is configured. This keeps the *cheap, fast* path local for routine cases.
    cloud_escalation_confidence: float = field(
        default_factory=lambda: _env_float("CLOUD_ESCALATION_CONFIDENCE", 65.0)
    )
    prefer_cloud: bool = field(default_factory=lambda: _env_bool("PREFER_CLOUD", False))

    # ---- Security Configuration ---------------------------------------------
    enable_auth: bool = field(default_factory=lambda: _env_bool("ENABLE_AUTH", True))
    # Relaxed-security DEV mode for the API auth layer. This is an EXPLICIT flag and
    # must NEVER be inferred from the compute device — a CPU/edge box is still
    # production, and inferring dev-mode from "device == cpu" silently disables the
    # JWT-secret validation on exactly the hardware this project targets.
    auth_dev_mode: bool = field(default_factory=lambda: _env_bool("AUTH_DEV_MODE", True))
    jwt_secret: str = field(
        default_factory=lambda: os.environ.get("JWT_SECRET", "dev-secret-key-change-in-production")
    )
    access_token_expire_minutes: int = field(
        default_factory=lambda: _env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    )
    refresh_token_expire_days: int = field(
        default_factory=lambda: _env_int("REFRESH_TOKEN_EXPIRE_DAYS", 7)
    )

    # OIDC Configuration
    oidc_enabled: bool = field(default_factory=lambda: _env_bool("OIDC_ENABLED", False))
    oidc_discovery_url: str = field(
        default_factory=lambda: os.environ.get("OIDC_DISCOVERY_URL", "")
    )
    oidc_client_id: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_ID", ""))
    oidc_client_secret: str = field(
        default_factory=lambda: os.environ.get("OIDC_CLIENT_SECRET", "")
    )

    # Rate Limiting Configuration
    enable_rate_limiting: bool = field(
        default_factory=lambda: _env_bool("ENABLE_RATE_LIMITING", True)
    )
    rate_limit_admin: str = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_ADMIN", "1000/minute")
    )
    rate_limit_clinician: str = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_CLINICIAN", "100/minute")
    )
    rate_limit_researcher: str = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_RESEARCHER", "50/minute")
    )
    rate_limit_default: str = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_DEFAULT", "20/minute")
    )

    # ---- Ethical guardrails ----------------------------------------------
    confidence_floor: float = field(default_factory=lambda: _env_float("CONFIDENCE_FLOOR", 60.0))

    # ---- OASIS-3 / OASIS-4 data access ------------------------------------
    nitrc_username: str = field(
        default_factory=lambda: os.environ.get("NITRC_USERNAME", "hakeemadeniji")
    )
    xnat_base_url: str = field(
        default_factory=lambda: os.environ.get("XNAT_BASE_URL", "https://central.xnat.org")
    )
    # Where FreeSurfer stats (aseg.stats / *.aparc.stats) land after extraction.
    freesurfer_root: str = field(
        default_factory=lambda: os.environ.get("FREESURFER_ROOT", "data/oasis_freesurfer")
    )
    oasis_scans_root: str = field(
        default_factory=lambda: os.environ.get("OASIS_SCANS_ROOT", "data/oasis_scans")
    )
    # PET Unified Pipeline (PUP) outputs -> amyloid/tau SUVR for the ATN A/T axes.
    pup_root: str = field(default_factory=lambda: os.environ.get("PUP_ROOT", "data/oasis_pup"))
    # Amyloid-positivity threshold on the Centiloid scale.
    amyloid_positive_centiloid: float = field(
        default_factory=lambda: _env_float("AMYLOID_POSITIVE_CENTILOID", 20.0)
    )
    # Tau-positivity threshold on meta-ROI SUVR (e.g. AV1451 temporal meta-ROI).
    tau_positive_suvr: float = field(default_factory=lambda: _env_float("TAU_POSITIVE_SUVR", 1.27))

    def resolve_device(self) -> str:
        """Resolve ``device='auto'`` to a concrete torch device string."""
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch  # local import keeps config import-cheap

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def cloud_available(self) -> bool:
        return bool(self.ollama_cloud_url.strip())

    def anthropic_available(self) -> bool:
        return self.enable_anthropic and bool(self.anthropic_api_key.strip())

    def summary(self) -> str:
        anth = (
            f"anthropic=on(cheap={self.anthropic_model_cheap},std={self.anthropic_model_standard},"
            f"deep={self.anthropic_model_deep})"
            if self.anthropic_available()
            else "anthropic=off"
        )
        return (
            f"device={self.resolve_device()} | onnx_providers={self.onnx_providers} | "
            f"ollama={'on' if self.enable_llm else 'off'}"
            f"(edge={self.ollama_edge_model}) | {anth}"
        )


@lru_cache(maxsize=1)
def get_settings() -> EdgeCloudSettings:
    """Return the process-wide settings singleton (cached)."""
    return EdgeCloudSettings()


if __name__ == "__main__":
    s = get_settings()
    print("OASIS Agentic Pipeline - resolved hybrid edge-cloud settings")
    print("-" * 60)
    print(s.summary())
    print(f"ARM64 detected      : {is_arm64()}")
    print(f"Preferred providers : {s.onnx_providers}")
    print(f"NITRC username      : {s.nitrc_username}")
