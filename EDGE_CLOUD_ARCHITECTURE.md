# Hybrid Edge-Cloud Architecture (Windows ARM64 / Snapdragon NPU)

This document describes how the OASIS Agentic Pipeline runs as a **hybrid
edge-cloud** system optimized for **Windows on ARM64 (Snapdragon X / Hexagon
NPU)** while using **only free, locally-hosted AI** — there are no metered API
keys anywhere in the pipeline.

The clinical goal is unchanged: explainable, ethically-guardrailed, multi-modal
Alzheimer's disease **screening decision-support** on the OASIS datasets.

---

## 1. Why this design

| Constraint | Decision |
|------------|----------|
| Runs on a Snapdragon NPU laptop | ONNX Runtime with **QNN (Hexagon NPU) → DirectML → CPU** auto-fallback |
| No paid API tokens | All LLM reasoning on **local Ollama** (open-weight models) |
| Must keep working offline at the edge | Deterministic **template fallback** when Ollama is unreachable |
| Real, deployable clinical signal | **Regional volumetry** from OASIS-3/4 FreeSurfer derivatives |
| Honest effectiveness measurement | Reproducible **PDF analysis** generated from a real evaluation |

---

## 2. Tiered topology

```
                ┌──────────────────────── EDGE (Snapdragon device) ─────────────────────────┐
                │                                                                            │
   MRI + CSV ──►│  Vision (ResNet18, ONNX INT8 on Hexagon NPU)   Biomarker / Temporal        │
                │  Regional Volumetry (FreeSurfer aseg z-scores)  RAG retrieval (local)       │
                │  Ethicist guardrail (deterministic)                                         │
                │                         │                                                  │
                │                         ▼                                                  │
                │     Clinical Reasoner (Agent 8) ── Ollama edge model (e.g. llama3.2:3b)     │
                │                         │                                                  │
                └─────────────────────────┼──────────────────────────────────────────────────┘
                                          │  escalate IF (confidence < floor) OR (ethicist flag)
                                          ▼
                ┌──────────────────────── CLOUD (optional, self-hosted) ────────────────────┐
                │     Ollama "cloud" endpoint with a larger open model (e.g. llama3.1:8b)     │
                │     Still free / no API key — just a bigger box you control.                │
                └────────────────────────────────────────────────────────────────────────────┘
```

* **Edge-first:** routine, high-confidence cases are fully resolved on-device.
* **Cloud-escalation:** only low-confidence or ethically-flagged cases are sent
  to the optional larger model. If no cloud endpoint is configured, everything
  stays on the edge.
* **Never hard-fails:** if no Ollama daemon is reachable at all, a deterministic
  templated narrative is produced so the device keeps operating offline.

Configuration lives in [`src/config.py`](src/config.py) and is driven entirely
by environment variables (see [`.env.example`](.env.example)).

---

## 3. NPU acceleration

[`src/utils/npu_runtime.py`](src/utils/npu_runtime.py) builds the execution
provider chain and reconciles it against the providers actually compiled into
the installed `onnxruntime`:

1. `QNNExecutionProvider` — Qualcomm Hexagon NPU (install `onnxruntime-qnn`).
2. `DmlExecutionProvider` — DirectML on the ARM64 GPU (`onnxruntime-directml`).
3. `CPUExecutionProvider` — universal fallback (always present).

The INT8-quantized multimodal fusion model
(`src/pipeline/onnx_inference/multimodal_fusion_int8.onnx`) is the artifact that
lights up the NPU. The same file runs on any tier; the runtime simply selects
the best available silicon and logs which one served inference (useful for fleet
telemetry). To enable the NPU:

```powershell
pip uninstall onnxruntime
pip install onnxruntime-qnn      # Snapdragon NPU
# or: pip install onnxruntime-directml   # ARM64 GPU
```

---

## 4. Local LLM reasoning (Ollama)

The **Clinical Reasoner** ([`src/agents/llm/llm_reasoner.py`](src/agents/llm/llm_reasoner.py))
is the only "generative" agent. It is *grounded*: it summarizes the structured,
already-computed outputs of the deterministic agents and is explicitly forbidden
from inventing a diagnosis. The authorized classification still comes from the
vision model gated by the ethicist.

```powershell
./scripts/setup_ollama.ps1          # installs Ollama + pulls llama3.2:3b
```

Suggested edge models for Snapdragon: `llama3.2:3b`, `phi3.5`, `qwen2.5:3b`.

---

## 5. Regional volumetry (real deployable signal)

[`src/agents/biomarker/volumetry_agent.py`](src/agents/biomarker/volumetry_agent.py)
parses FreeSurfer `aseg.stats`, normalizes each structure by eTIV, z-scores
against an elderly normative reference, and produces a medial-temporal-atrophy
(MTA) risk score. This moves the system beyond a single whole-brain `nWBV`
number toward the regional measurements clinicians actually reason about
(hippocampus, amygdala, ventricles).

It consumes the **OASIS-3 / OASIS-4 FreeSurfer** derivatives downloaded by the
scripts in [`scripts/oasis/`](scripts/oasis/). Point `FREESURFER_ROOT` at the
output directory. When no FreeSurfer derivative exists (e.g. a 2D OASIS-1
slice), the agent degrades gracefully to a whole-brain estimate.

---

## 6. OASIS-3 / OASIS-4 ingestion

[`scripts/oasis/`](scripts/oasis/) vendors the official NITRC/XNAT download
scripts (scans, FreeSurfer, PUP, partial-match) plus a Windows ARM64 PowerShell
wrapper. NITRC username defaults to `hakeemadeniji`. See
[`scripts/oasis/README.md`](scripts/oasis/README.md).

---

## 7. Effectiveness analysis

[`src/pipeline/evaluation/effectiveness_report.py`](src/pipeline/evaluation/effectiveness_report.py)
runs a real evaluation (vision metrics, cohort/longitudinal analysis, guardrail
battery, volumetry separation, per-provider latency benchmark) and renders
`docs/OASIS_Pipeline_Effectiveness_Analysis.pdf` using Pillow (no matplotlib
dependency, so it builds on win-arm64).

```bash
python src/pipeline/evaluation/effectiveness_report.py --max-per-class 60
```
