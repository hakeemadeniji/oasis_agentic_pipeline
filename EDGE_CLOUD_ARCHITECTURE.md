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
| Cost-effective agentic reasoning | **Hybrid LLM**: free local **Ollama** for tasks it does as well; **Claude** (`ANTHROPIC_API_KEY`, cost-tiered Haiku→Sonnet→Opus) for the hard reasoning |
| Must keep working offline at the edge | Graceful fallback **Claude → Ollama → deterministic template** |
| Real, deployable clinical signal | **Regional volumetry** + **ATN staging** from OASIS-3/4 FreeSurfer/PUP |
| Find treatment/cure clues | **Cure-research engine** (deterministic mining) + **Therapeutic Insight agent** (Claude) |
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

## 4. Hybrid LLM reasoning (Ollama + Claude)

[`src/agents/llm/llm_provider.py`](src/agents/llm/llm_provider.py) is the routing
layer. It sends each task to the cheapest capable backend and degrades
gracefully (**Claude → Ollama → deterministic template**):

| Task | Tier | Backend |
|------|------|---------|
| Grounded summaries, structured labels, routine narration | free | local **Ollama** |
| Short structured extraction | cheap | **Claude Haiku 4.5** |
| Clinical synthesis (flagged/low-confidence narration) | standard | **Claude Sonnet 4.6** |
| Differential diagnosis (Agent 11), cure-research hypotheses (Agent 12) | deep | **Claude Opus 4.8** (adaptive thinking) |

All generative agents are *grounded*: they reason over the structured outputs of
the deterministic agents and never invent the authorized classification. Set
`ANTHROPIC_API_KEY` to enable the Claude tiers; `PREFER_FREE_WHEN_CAPABLE=true`
keeps routine work on free Ollama. Batch API (50% off) is available for bulk
cohort work.

```powershell
./scripts/setup_ollama.ps1          # free local tier: Ollama + llama3.2:3b
# set ANTHROPIC_API_KEY in .env to enable the Claude tiers
```

New reasoning agents:
- **Differential Diagnosis** ([`differential_agent.py`](src/agents/llm/differential_agent.py)) —
  ranked etiologies (AD / FTD / DLB / vascular / normal-aging) + work-up, with a
  deterministic rule-based prior as both grounding and offline fallback.
- **Therapeutic Insight / Cure-Research** ([`therapeutic_agent.py`](src/agents/llm/therapeutic_agent.py)) —
  reasons over [`pipeline/research/cure_research.py`](src/pipeline/research/cure_research.py)
  (deterministic associations + protective-factor / target mining over OASIS) to
  generate **testable** research hypotheses. Exposed at `/research`.

Suggested free edge models for Snapdragon: `llama3.2:3b`, `phi3.5`, `qwen2.5:3b`.

## 4b. Interactive brain map

The console's **Brain Map** view renders an anatomical SVG whose regions glow
with pathology severity (hippocampus/amygdala atrophy, ventricular enlargement)
driven by the regional-volumetry z-scores — hover any region for its z-score.

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
