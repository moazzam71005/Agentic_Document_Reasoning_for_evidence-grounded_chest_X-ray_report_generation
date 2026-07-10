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
