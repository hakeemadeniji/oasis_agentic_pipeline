# Advanced Diagnostic Analytics — What the OASIS Data Can Support

This document surveys the diagnostic analyses that are feasible with the data
this project now has access to (**OASIS-1**, plus **OASIS-3 / OASIS-4** via the
NITRC download scripts), ordered from what is already implemented to the most
advanced, research-grade analyses. Each entry notes the **required data**, the
**method**, and the **implementation status** in this repo.

> Scope note: this is *screening / research decision-support*, not a diagnostic
> device. Every advanced analytic below is designed to feed the ethicist
> guardrail and route uncertain/positive findings to a clinician.

---

## 1. Data inventory (what's on hand)

| Source | Modality | Key variables usable for analytics |
|--------|----------|-------------------------------------|
| **OASIS-1** (bundled) | 2D T1 MRI slices (4 CDR classes) | per-slice severity class, clinical CSV (Age, MMSE, eTIV, nWBV, ASF, Educ, SES, M/F) |
| **OASIS-1 longitudinal** | repeat visits | MR Delay, CDR, nWBV trajectory |
| **OASIS-3** | 3D T1/T2/FLAIR, **FreeSurfer**, **PUP PET** (PIB & AV45 amyloid, **AV1451 tau**), FDG | regional volumes/thickness, **SUVR / Centiloid**, longitudinal sessions, ADRC clinical (CDR, MMSE, APOE) |
| **OASIS-4** | memory-clinic multimodal | structural MRI + clinical workup for differential dementia |

The download scripts in [`scripts/oasis/`](scripts/oasis/) fetch scans,
FreeSurfer derivatives (`stats/aseg.stats`, `*.aparc.stats`), and PUP outputs
(SUVR by ROI) that unlock everything in §3–§4.

---

## 2. Implemented today

| Analysis | Method | Module |
|----------|--------|--------|
| MRI severity classification | ImageNet-pretrained ResNet18 (NPU INT8), 4-class CDR proxy; **~37% balanced acc on a subject-disjoint held-out test set** (chance 25%; research-only — patient-poor data, see README Validation Status) | `agents/vision/` |
| Explainability | Grad-CAM activation mapping over MRI | `agents/vision/explainer_agent.py` |
| Whole-brain + **regional volumetry** | FreeSurfer `aseg` → eTIV-normalized **z-scores** + medial-temporal-atrophy (MTA) score | `agents/biomarker/volumetry_agent.py` |
| Longitudinal atrophy velocity | nWBV change / yr, MMSE drift, trend classification | `agents/biomarker/temporal_analyst.py` |
| **ATN biomarker profiling** | Amyloid (Centiloid) / Tau (SUVR) / Neurodegeneration → NIA-AA biological category | `agents/biomarker/atn_classifier.py` |
| Cross-modal safety audit | rule-based contradiction/uncertainty guardrails | `orchestrator/ethicist_agent.py` |
| Evidence retrieval + reasoning | local embeddings RAG + Ollama narrator | `agents/rag/`, `agents/llm/` |

---

## 3. High-value analyses unlocked by OASIS-3/4 (feasible now)

### 3.1 ATN biomarker classification (NIA-AA 2018 research framework) — *implemented*
- **A (Amyloid):** PUP **mean cortical SUVR (MCSUVR)** from precuneus, prefrontal
  cortex, gyrus rectus, lateral temporal (cerebellar reference) → **Centiloid**;
  amyloid-positive at ≈ **+20 CL**.
- **T (Tau):** AV1451 **tau SUVR** in an entorhinal/temporal meta-ROI → T+/-.
- **N (Neurodegeneration):** hippocampal/cortical atrophy z-score (our volumetry
  agent) or FDG hypometabolism.
- **Output:** A±T±N± profile → biological categories (A-T-N- normal; A+T-N-
  *Alzheimer's pathologic change*; A+T+ *Alzheimer's disease*; A-T+/N+ *non-AD
  pathologic change* / SNAP). This is the single most clinically meaningful
  upgrade and is now in `atn_classifier.py`.

### 3.2 Centiloid harmonization
Convert tracer-specific SUVR (PIB vs AV45) to the **Centiloid** scale so amyloid
burden is comparable across tracers/sites. Implemented as a tracer-aware linear
transform in the ATN module (coefficients are calibration-dependent priors).

### 3.3 Tau topographic (Braak-like) staging
Map AV1451 SUVR across FreeSurfer ROIs to **Braak-stage regions** (entorhinal →
limbic → neocortical) to estimate tau spread stage. Requires regional PUP tau
SUVR; staging logic is a natural extension of the ATN module.

### 3.4 BrainAGE (brain-predicted age gap)
Train a regressor on structural features (regional volumes/thickness, eTIV,
nWBV) of cognitively-normal subjects to predict chronological age; the
**predicted-minus-actual gap** is a sensitive neurodegeneration marker. Feasible
with OASIS-3 FreeSurfer + ages. (Roadmap: `pipeline/brain_age.py`.)

### 3.5 Normative modeling / W-scores
Replace point z-scores with **age/sex/eTIV-adjusted normative percentiles**
(W-scores or Gaussian-process normative models) fit on the cognitively-normal
OASIS-3 cohort, giving each region a personalized deviation map.

### 3.6 Progression / conversion modeling
With longitudinal OASIS-3 (CDR, MMSE, volumes over time):
- **Survival / time-to-conversion** (Cox / discrete-time) for CN→MCI→AD.
- **Mixed-effects trajectories** of atrophy and cognition.
- **Disease-progression scores** (e.g., event-based / SuStaIn-style ordering of
  biomarker abnormality) — research-grade subtype + stage inference.

---

## 4. Most-advanced (research-grade) directions

- **3D volumetric CNN / hippocampal-subfield models** on full OASIS-3 T1 volumes
  (vs the current 2D slice model) with NPU-exported INT8 graphs.
- **Multimodal fusion** of MRI + amyloid + tau + clinical via cross-attention
  (the `fusion_agent.py` scaffold) for an end-to-end risk score.
- **Multimodal foundation embeddings**: self-supervised representations of
  scans + tabular biomarkers, enabling clustering into **data-driven AD
  subtypes** (typical / limbic-predominant / hippocampal-sparing).
- **Uncertainty quantification & calibration** (deep ensembles / MC-dropout,
  temperature scaling) so the ethicist can gate on *calibrated* confidence.
- **Differential dementia** (OASIS-4): multi-class beyond AD (FTD, DLB, vascular)
  using the memory-clinic workup.
- **Federated / on-device continual learning** across edge sites (privacy-
  preserving), aligned with the Snapdragon-NPU edge deployment.

---

## 5. Recommended next implementation order

1. **ATN + Centiloid** (done) — highest clinical signal per unit effort.
2. **BrainAGE** regressor on OASIS-3 FreeSurfer features.
3. **Normative W-scores** for the volumetry agent.
4. **Conversion/survival** model on longitudinal OASIS-3.
5. **3D volumetric model** + multimodal fusion retrain, INT8-exported for NPU.

---

## Sources
- [OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset (medRxiv)](https://www.medrxiv.org/content/10.1101/2019.12.13.19014902v1.full)
- [OASIS-3 Imaging Methods & Data Dictionary (WUSTL)](https://bpb-us-e2.wpmucdn.com/sites.wustl.edu/dist/6/4383/files/2024/04/OASIS-3_Imaging_Data_Dictionary_v2.3-a93c947a586e7367.pdf)
- [petBrain: amyloid, tau, and neurodegeneration quantification using PET and MRI (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12482854/)
- [DeepSUVR: temporal constraints for SUVR and Centiloid quantification (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11713578/)
- [Aβ Centiloid pipeline comparisons across cohorts (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12740321/)
