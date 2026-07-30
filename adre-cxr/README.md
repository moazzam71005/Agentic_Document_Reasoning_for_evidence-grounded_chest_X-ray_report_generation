# ADRE-CXR

Agentic Document Reasoning for evidence-grounded chest X-ray report generation, adapted from the ADRE EEG paper (Masood et al.).

**No model training required.** Uses pretrained vision models + cloud LLM APIs + RAG.

## Quick start (no local GPU)

### 1. Install

```bash
cd adre-cxr
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env
```

### 2. Get a free LLM API key (pick one)

| Provider | Cost | Sign up | Recommended model |
|---|---|---|---|
| **Groq** | Free tier | [console.groq.com](https://console.groq.com/keys) | `llama-3.3-70b-versatile` |
| **Google Gemini** | Free tier | [aistudio.google.com](https://aistudio.google.com/apikey) | `gemini-2.0-flash` |
| **HuggingFace** | Free, rate-limited | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `Qwen/Qwen2.5-7B-Instruct` |

Edit `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Groq is recommended — fast, free, runs entirely in the cloud. Your laptop only needs CPU and internet.

### 3. Get IU X-Ray data

**Option A — Kaggle (recommended):**

1. Download [raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
2. Unzip to `data/raw/chest-xrays-indiana-university/`

**Option B — Kaggle CLI:**

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
python -c "from src.data_prep.download import download_iu_xray; download_iu_xray(__import__('pathlib').Path('data/raw'))"
```

### 4. Prepare splits

```bash
python scripts/prepare_data.py --data-root data/raw/chest-xrays-indiana-university
```

### 5. Evidence extraction (Kaggle GPU — optional but better)

The vision classifier needs a GPU. Use **Kaggle's free GPU** (~30 hrs/week), not your local machine:

1. Upload this repo + open `notebooks/01_kaggle_evidence_extraction.ipynb`
2. Attach dataset `raddar/chest-xrays-indiana-university`
3. Enable GPU, run all cells
4. Download `evidence/` folder → place in `data/processed/evidence/`

**CPU fallback** (no pathology probabilities, but pipeline still runs):

```bash
python scripts/extract_evidence.py --data-root data/raw/chest-xrays-indiana-university --split dev --skip-neural
```

### Evidence v2 (recommended experimental extractor)

Evidence v2 changes only the visual evidence layer. Report prompts, narrative
generation, and report metrics are unchanged. It processes every available IU
projection, validates image quality, preserves aspect ratio, keeps original and
CLAHE branches, records uncalibrated model scores honestly, assigns
positive/negative/uncertain/not-assessable states, generates Grad-CAM
model-attention regions, uses explicitly heuristic anatomy masks, fuses views,
scores reliability, and writes audit artifacts.

The default configuration is `configs/evidence_v2.yaml`. IU-Xray does not
contain reliable pathology labels for fitting a probability calibrator, so the
default output says `confidence_type: heuristic_model_score` and
`calibrated: false`. Do not describe these scores as clinical probabilities.

Run on Kaggle from the repository root:

```bash
pip install -r requirements.txt

# Existing frozen ADRE-style split: 100 development studies
python scripts/extract_evidence_v2.py \
  --data-root /kaggle/input/chest-xrays-indiana-university \
  --split dev

# Existing frozen ADRE-style split: 400 test studies
python scripts/extract_evidence_v2.py \
  --data-root /kaggle/input/chest-xrays-indiana-university \
  --split test
```

If Kaggle mounted the dataset under a longer path, use the directory that
directly contains `indiana_reports.csv` and `indiana_projections.csv`.

Each study is stored at:

```text
data/processed/evidence_v2/<uid>/
  evidence.json
  evidence_v1_compatible.json
  prediction_summary.png
  *_normalized.png
  *_clahe.png
  anatomy/
  gradcam/
  crops/
  embeddings/
```

Validate before generating any reports:

```bash
python scripts/validate_evidence.py data/processed/evidence_v2
python scripts/inspect_evidence.py data/processed/evidence_v2/<uid>/evidence.json
python scripts/compare_evidence_versions.py \
  --v1-dir data/processed/evidence \
  --v2-dir data/processed/evidence_v2 \
  --output outputs/evidence_version_comparison.json
```

The existing report pipeline can load `evidence.json` directly; the evidence
loader converts it to the legacy packet in memory:

```bash
python scripts/run_pipeline.py \
  --study-id <uid> \
  --image <frontal-image-path> \
  --evidence-json data/processed/evidence_v2/<uid>/evidence.json \
  --output outputs/<uid>.txt
```

`run_batch.py` also recognizes the nested v2 directory automatically:

```bash
python scripts/run_batch.py \
  --data-root data/raw/chest-xrays-indiana-university \
  --split dev \
  --limit 100 \
  --evidence-dir data/processed/evidence_v2 \
  --output-dir outputs/evidence_v2_dev
```

Before LLM report evaluation, create and manually verify a 50-100 study
evidence-review file:

```bash
python scripts/create_evidence_review_set.py \
  --split-file data/processed/dev_split.json \
  --evidence-dir data/processed/evidence_v2 \
  --size 75 \
  --output outputs/evidence_review_set.jsonl

# Fill the review fields and set verified to true after manual review.
python scripts/evaluate_evidence.py outputs/evidence_review_set.jsonl
```

Controlled evidence ablations use the same split and configuration:

```bash
python scripts/extract_evidence_v2.py \
  --data-root <iu-dataset-root> \
  --split dev \
  --ablation 1 \
  --output-dir data/processed/evidence_v2_ablation_1
```

Repeat `--ablation 1` through `--ablation 7`. Ablation 7 enables the optional
second TorchXRayVision model configured in the YAML. Cache keys include image
checksums, model versions, configuration hash, and extractor version; use
`--force` only when intentionally rebuilding a matching result.

### 6. Guideline corpus (manual)

Download and place PDFs in `data/guidelines/`:

- [Fleischner Society Glossary](https://fleischner.org/wp-content/uploads/2020/12/Fleischner-Society-Glossary-of-Terms-for-Thoracic-Imaging.pdf)

Build retrieval index (CPU only, ~2 min):

```python
from pathlib import Path
from src.knowledge_index.parse_guidelines import parse_guideline_dir
from src.knowledge_index.build_index import build_index

chunks = parse_guideline_dir(Path("data/guidelines"))
build_index(chunks, Path("data/processed/index"))
```

### 7. Run pipeline on one study

```bash
python scripts/run_pipeline.py ^
  --study-id 1 ^
  --image "data/raw/chest-xrays-indiana-university/images/images_normalized/1_1.png" ^
  --indication "chest pain" ^
  --evidence-json "data/processed/evidence/1.json" ^
  --output output_report.txt
```

Report appears in `output_report.txt`.

---

## Architecture (matches ADRE paper)

```
Chest X-ray → Evidence packet (pretrained classifier + zone features)
           → Agentic loop (gap analysis + guideline retrieval)
           → Stage A: structured findings JSON (T=0.0)
           → Stage B: Findings + Impression narrative (T=0.25)
```

## Where each piece runs

| Component | Where | GPU needed? |
|---|---|---|
| LLM (reasoning + generation) | **Groq/Gemini cloud API** | No |
| Vision classifier (TorchXRayVision) | **Kaggle notebook** | Yes (free) |
| Retrieval index (BM25 + FAISS) | Local CPU or Kaggle CPU | No |
| Evaluation (BERTScore) | Kaggle GPU or local CPU | Optional |

## Cloud vs local summary

```
YOUR LAPTOP (CPU only)          KAGGLE (free GPU, 30 hrs/week)
─────────────────────          ────────────────────────────────
• Groq/Gemini LLM calls        • Batch evidence extraction
• Agentic loop                 • BERTScore evaluation
• Report generation            • Data prep at scale
• Retrieval index build
```

You never need a local GPU.

## Project structure

See `plan.md` in the repo root for the full implementation roadmap and ablation design.

## License note

IU X-Ray Kaggle mirror is **CC BY-NC-ND 4.0** — academic use OK, check NC/ND restrictions.
