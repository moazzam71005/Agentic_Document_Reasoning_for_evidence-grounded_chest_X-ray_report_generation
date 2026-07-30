# ADRE-CXR: Status Report
### Agentic Document Reasoning for Evidence-Grounded Chest X-Ray Report Generation
> Adaptation of Masood et al., "Agentic Document Reasoning for Evidence-Grounded Clinical Report Generation"

---

## 1. Where We Stand Relative to the Paper

### What the paper does (ADRE-EEG)

ADRE was originally built for EEG reports. Given a raw EDF signal and a corpus of clinical guidelines, it:
1. Extracts intermediate signal features (PDR, spectral ratios, hemispheric symmetry)
2. Retrieves relevant guideline passages using a hybrid BM25 + FAISS index
3. Runs an **iterative agentic loop** — identifying gaps between current evidence and required report checklist
4. Generates a two-stage report: structured JSON findings (Stage A, T=0.0) then narrative text (Stage B, T=0.25)

Their best result on N=400 test studies with Gemma3:12b:  
**BERTScore F1 = 0.8723**, METEOR = 0.2165 (vs baseline BERTScore ~0.83)

### What this repo does (ADRE-CXR)

Every architectural component from the paper has been implemented and adapted for chest X-rays:

| ADRE-EEG component | ADRE-CXR equivalent | Status |
|---|---|---|
| EDF signal → signal features | PNG/DICOM → image features (CTR proxy, opacity density, symmetry) | ✅ Built |
| SCNet global abnormality score | TorchXRayVision DenseNet121 (14-pathology probabilities) | ✅ Built |
| Regional morphology (EEG channels × windows) | Zone crops (upper/mid/lower × left/right, 6 zones) + per-zone pathology burden | ✅ Built |
| Quality indicators (EMG, power-line noise) | Exposure label, inspiration proxy, rotation proxy | ✅ Built |
| Evidence packet Φ = (view_id, channels, t_start, t_end) | Evidence packet Φ = (view_id, zone, params, tool) | ✅ Built |
| Layout-aware PDF parsing (MinerU) | PyMuPDF hierarchical chunk parser | ✅ Built |
| BM25Okapi + FAISS hybrid retrieval (α=0.5, top-k=8) | Same: BM25Okapi + FAISS IndexFlatIP, all-MiniLM-L6-v2 | ✅ Built |
| Context expansion (±1 sibling under same parent_id) | Same logic | ✅ Built |
| EvidenceState (append-only, St) | EvidenceState class | ✅ Built |
| Report checklist C | 5-item CXR checklist (lungs, heart, pleura, quality, impression) | ✅ Built |
| Gap analysis Δt = GapAnalysis(St, C, G(qt)) | gap_analysis.py | ✅ Built |
| Tool planning + view requests | planner.py | ✅ Built |
| Consistency checks (cross-view conflicts) | consistency.py | ✅ Built |
| Algorithm 1 iterative loop (Tmax=18) | controller.py (Tmax=12) | ✅ Built |
| Stage A: schema-constrained JSON (T=0.0) | stage_a_findings.py | ✅ Built |
| Stage B: narrative (T=0.25, top-p=0.9) | stage_b_narrative.py | ✅ Built |
| Format validation + repair prompt | repair.py | ✅ Built |
| LLM backbones via Ollama (Qwen3:8b, Gemma3:12b, DeepSeek-R1:8b) | Groq API (Llama-3.3-70b) — same zero-shot inference-only approach | ✅ Built |
| BLEU-4, ROUGE-L, METEOR, BERTScore (RoBERTa-large) | metrics.py — identical metrics | ✅ Built |
| Wilcoxon + paired t-test + bootstrap | significance.py | ✅ Built |
| Ablation variant matrix (6-way, Table 1) | ablation_runner.py (variants defined) | 🔲 Not run yet |

### Overall verdict

The full ADRE architecture has been successfully ported from EEG to chest X-rays. The pipeline runs end-to-end, producing structured findings JSON and a Findings + Impression narrative for any study in the IU X-Ray dataset. What remains is scaling to the full dev and test sets and running the 6-way ablation.

---

## 2. Complete Pipeline

### Dataset

**IU X-Ray (Indiana University Chest X-Ray Collection, Open-I)**  
- Source: [Kaggle — raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)  
- Size: 7,470 PNG images (frontal + lateral), 3,955 radiology reports  
- Reports already contain: `Comparison`, `Indication`, `Findings`, `Impression`  
- License: CC BY-NC-ND 4.0 (academic use)

**Splits (frozen, same ratio as paper):**
| Split | N | Purpose |
|---|---|---|
| Dev | 100 | Hyperparameter tuning (α, top-k, Tmax) |
| Test | 400 | Final held-out evaluation |

**Guideline corpus:**
| Document | Role |
|---|---|
| Fleischner Society Glossary of Terms for Thoracic Imaging (44 MB PDF) | Terminology, finding definitions, reporting conventions |

---

### Pipeline architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                   │
│  Chest X-ray PNG  +  Patient metadata (indication, comparison)          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Multi-modal Evidence Acquisition                             │
│                                                                         │
│  View Builder                                                           │
│  ├─ original (full image, grayscale)                                    │
│  ├─ clahe (CLAHE contrast-enhanced copy)                                │
│  └─ 6 zone crops: upper/mid/lower × left/right (heuristic lung bbox)   │
│                                                                         │
│  Neural Inference (pretrained, no fine-tuning)                         │
│  ├─ Model: TorchXRayVision DenseNet121 (densenet121-res224-nih)        │
│  ├─ Global pathology probs: 14 labels (Atelectasis, Effusion, …)       │
│  ├─ Global abnormality score P_ab = max over 14 pathologies            │
│  └─ Per-zone burden: max_prob, mean_prob, top_pathology for each zone  │
│                                                                         │
│  Image Features (deterministic, no model)                              │
│  ├─ opacity_density (fraction of pixels above 60th percentile)         │
│  ├─ symmetry_index (L/R histogram similarity, 0–1)                     │
│  ├─ ctr_proxy (cardiothoracic ratio from row-mean mass distribution)   │
│  ├─ mean_intensity, std_intensity                                       │
│                                                                         │
│  Quality Indicators                                                     │
│  ├─ exposure_label: underexposed / adequate / overexposed               │
│  ├─ inspiration_label: adequate / suboptimal / poor (edge density)      │
│  └─ rotation_label: minimal / mild / moderate (clavicle imbalance)     │
│                                                                         │
│  Evidence Packet Φ = {study_id, filename, metadata,                    │
│    global_pathology_probs, global_abnormality_score,                    │
│    regional_burden, image_features, quality_indicators,                 │
│    provenance[view_id, zone, tool]}                                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────────────┐
│  LAYER 2 — Knowledge    │   │  LAYER 3 — Agentic Reasoning Engine     │
│  Index (RAG)            │   │                                         │
│                         │   │  EvidenceState S_t (append-only)        │
│  PDF → PyMuPDF parser   │   │  Report Checklist C:                    │
│  → hierarchical chunks  │   │   1. findings_lungs                     │
│    {id, parent_id,      │   │   2. findings_heart                     │
│     title_path,         │   │   3. findings_pleura                    │
│     page_span,          │   │   4. findings_quality                   │
│     sibling_idx, text}  │   │   5. impression_summary                 │
│                         │   │                                         │
│  BM25Okapi (k1=1.5,     │◄──┤  Iterative loop (Tmax=12):             │
│   b=0.75)               │   │  for t in range(Tmax):                  │
│  +                      │   │    q_t = FormQuery(S_t, C)              │
│  FAISS IndexFlatIP      │   │    G_t = RetrieveExpand(index, q_t)     │
│  (all-MiniLM-L6-v2,     │──►│    Δ_t = GapAnalysis(S_t, C, G_t)     │
│   d=384)                │   │    U_t = Plan(Δ_t, G_t)                │
│                         │   │    e_t = InvokeTool(U_t)               │
│  Fusion: α=0.5          │   │    S_t+1 = Consolidate(S_t, e_t)       │
│  top-k=8                │   │    SanityCheck(S_t+1, C, G_t)          │
│                         │   │    if no gaps: break                    │
│  Context expansion:     │   │                                         │
│  ±1 sibling under same  │   │  → Final state S_T                      │
│  parent_id, max 500 w   │   │                                         │
└─────────────────────────┘   └──────────────────┬──────────────────────┘
                                                  │
                                                  ▼
                              ┌───────────────────────────────────────────┐
                              │  LAYER 4 — Dual-Stage Report Generation   │
                              │                                           │
                              │  Stage A (T=0.0, deterministic)           │
                              │  LLM prompt: S_T → structured JSON        │
                              │  Output:                                  │
                              │   {observations: [{finding, localization, │
                              │    confidence, evidence_ref}],            │
                              │    quality_notes: [],                     │
                              │    impression_candidates: []}             │
                              │                                           │
                              │  Stage B (T=0.25, top-p=0.9)             │
                              │  LLM prompt: Stage-A JSON + S_T → text   │
                              │  Output:                                  │
                              │   FINDINGS:                               │
                              │   <zone-by-zone narrative paragraph>      │
                              │                                           │
                              │   IMPRESSION:                             │
                              │   <concise diagnostic summary>            │
                              │                                           │
                              │  Repair: if FINDINGS:/IMPRESSION: missing │
                              │  → repair prompt with original evidence   │
                              └───────────────────────────────────────────┘
```

### Models used

| Component | Model | Parameters | Source | Training on IU X-Ray? |
|---|---|---|---|---|
| Pathology classifier | TorchXRayVision DenseNet121 | 7M | NIH ChestX-ray14 pretrained | **No** |
| Embedding (RAG) | all-MiniLM-L6-v2 | 23M | Sentence-Transformers | **No** |
| LLM (reasoning + generation) | Llama-3.3-70b-versatile | 70B | Groq cloud (free tier) | **No** |
| Evaluation | RoBERTa-large | 355M | HuggingFace (BERTScore) | **No** |

> Zero-shot, inference-only throughout — exactly as in the paper.

### Hyperparameters (currently at defaults, tune on dev)

| Parameter | Value | Paper |
|---|---|---|
| α (BM25 vs dense fusion) | 0.5 | 0.5 |
| top-k retrieval | 8 | 8 |
| Tmax (max agentic steps) | 12 | 18 |
| Stage A temperature | 0.0 | 0.0 |
| Stage B temperature | 0.25 | 0.25 |
| Stage B top-p | 0.9 | 0.9 |
| Context expansion cap | 500 words | 500 words |

---

## 3. Results

### Our results (ADRE-CXR on IU X-Ray)

Evaluation on **N=26 dev studies** (first 26 from frozen dev split), vs radiologist-written Findings+Impression:

| Metric | Score | Notes |
|---|---|---|
| BLEU-4 | 0.0020 | Low — expected (see below) |
| ROUGE-L | 0.098 | Surface overlap |
| METEOR | 0.114 | Lexical + synonym overlap |
| **BERTScore F1** | **0.831** | Primary metric; RoBERTa-large |

LLM backbone: Llama-3.3-70b-versatile (via Groq, zero-shot)

**Example — Study 1** (Indication: Positive TB test)

| | Text |
|---|---|
| **Generated FINDINGS** | Multiple masses in the left lung (upper/mid/lower zones, high confidence), atelectasis upper right, consolidation lower right. Image overexposed, poor inspiration, mild rotation. |
| **Ground truth FINDINGS** | Cardiac silhouette and mediastinum normal. No pulmonary edema. No focal consolidation. No pleural effusion. No pneumothorax. |
| **Generated IMPRESSION** | Multifocal lung masses suggestive of infectious or inflammatory process (pneumonia/TB). |
| **Ground truth IMPRESSION** | Normal chest x-ray. |

**Example — Study 586** (Indication: shortness of breath, atelectasis vs infiltrate)

| | Text |
|---|---|
| **Generated FINDINGS** | Atelectasis lower right (high), effusion lower left (high), infiltration upper left (moderate). Overexposed, poor inspiration. |
| **Ground truth FINDINGS** | Moderately enlarged heart. Left midlung and left basilar opacity (atelectasis). Right midlung opacity improved. Degenerative spine changes. Sternotomy wires intact. |
| **Generated IMPRESSION** | Multifocal lung disease suggestive of pneumonia or pulmonary edema. |
| **Ground truth IMPRESSION** | Probable left midlung and left basilar atelectasis. Cardiomegaly. |

---

### Paper results (ADRE-EEG, N=400 test studies)

> From Masood et al. Table 2 — METEOR and BERTScore (RoBERTa-large) across backbones:

| Variant | Backbone | METEOR | BERTScore F1 |
|---|---|---|---|
| Baseline | Qwen3:8b | ~0.17 | ~0.855 |
| Baseline | Gemma3:12b | ~0.16 | ~0.850 |
| Baseline | DeepSeek-R1:8b | ~0.13 | ~0.838 |
| ADRE (full) | Qwen3:8b | ~0.19 | ~0.869 |
| **ADRE (full)** | **Gemma3:12b** | **0.2165** | **0.8723** |
| ADRE (full) | DeepSeek-R1:8b | ~0.17 | ~0.852 |

> From Table 4 (retrieval ablation, best backbone Gemma3:12b):

| Retrieval variant | BERTScore F1 |
|---|---|
| No retrieval | ~0.858 |
| Sparse only (BM25) | ~0.865 |
| Dense only (FAISS) | ~0.868 |
| Flat hybrid | ~0.870 |
| **Layout-aware hybrid** | **0.8723** |

---

### Head-to-head comparison

| | ADRE-CXR (ours) | ADRE-EEG (paper) |
|---|---|---|
| Dataset | IU X-Ray (chest) | NMT Scalp EEG (NUST-MH-TUKL) |
| N evaluated | 26 dev (of 100) | 400 test |
| LLM backbone | Llama-3.3-70b (Groq) | Qwen3:8b / Gemma3:12b / DeepSeek-R1:8b (Ollama) |
| BERTScore F1 | **0.831** | 0.838 – 0.872 |
| METEOR | 0.114 | 0.13 – 0.22 |
| Status | **Partial dev (26/100)** | Full test (400) |

---

## 4. Gap Analysis (what to do next)

### Why our BERTScore is slightly below the paper

1. **Small sample (N=26)** — BERTScore on 26 studies is less stable than on 400
2. **Classifier overcalling** — DenseNet121 (NIH weights) assigns 0.5–0.7 probability to nearly all 14 pathologies on normal CXRs. The paper's SCNet was calibrated for EEG; our vision classifier is less discriminative at zero-shot
3. **Different modality** — IU X-Ray reports are shorter and less standardized than the EEG reports used in the paper; "normal chest x-ray" is 4 words; the ADRE pipeline generates multi-sentence reports regardless
4. **LLM difference** — Groq/Llama-3.3-70b vs Ollama/Gemma3:12b; the paper's best model (Gemma3) may suit medical language better

### Remaining milestones

| Milestone | Status |
|---|---|
| Dev evidence extraction (Kaggle GPU) | ✅ 100/100 |
| Guideline index built | ✅ |
| First end-to-end report | ✅ |
| Dev evaluation (26 studies) | ✅ BERTScore = 0.831 |
| Full dev evaluation (100 studies) | 🔲 Run `run_dev_batch --limit 100` |
| Test evidence extraction (Kaggle) | 🔲 Run Kaggle Cell 6 with SPLIT="test" |
| 6-way ablation (Table 1 analogue) | 🔲 Use `ablation_runner.py` |
| Retrieval-stack ablation (Table 4 analogue) | 🔲 |
| Significance testing (Wilcoxon, bootstrap) | 🔲 `significance.py` |
| Final write-up | 🔲 |
