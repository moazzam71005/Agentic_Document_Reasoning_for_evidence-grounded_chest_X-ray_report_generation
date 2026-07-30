# How to Run ADRE-LoRA (Evidence-Grounded LoRA)

> This is the proper ADRE-inspired LoRA — the model is trained with image evidence + guidelines,
> not just plain indication text.

---

## What You Need

- A Kaggle account (free)
- GPU enabled on Kaggle (T4 or P100 — free tier)
- Internet ON in the Kaggle notebook
- The IU X-Ray dataset already on Kaggle

---

## Step-by-Step Instructions

### Step 1 — Upload the Notebook to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) and sign in
2. Click **"Create"** → **"New Notebook"**
3. In the new notebook page, click the **three dots (⋯)** at the top right → **"Import Notebook"**
4. Upload the file: `adre-cxr/notebooks/03_adre_lora_finetune.ipynb`

---

### Step 2 — Attach the Dataset

1. On the right sidebar, click **"Add data"**
2. Search for: `raddar/chest-xrays-indiana-university`
3. Click **"Add"**

The notebook will automatically find it — no path changes needed.

---

### Step 3 — Enable GPU and Internet

On the right sidebar:
- **Accelerator**: Select `GPU T4 x1` (or P100 if available)
- **Internet**: Toggle **ON**

---

### Step 4 — Run the Notebook

Click **"Run All"** (or Shift+Enter through each cell).

**What each cell does:**

| Cell | What it does | Time |
|---|---|---|
| 1 | Install Python packages | ~3 min |
| 2 | Find dataset path | instant |
| 3 | Parse all IU X-Ray reports, create 90/10 train/eval split | ~30 sec |
| 4 | Define DenseNet121 classifier | instant |
| 5 | Define image feature helpers | instant |
| 6 | **Extract evidence** for ALL studies (DenseNet + zone crops + features) | ~20–40 min |
| 7 | Build inline radiology guidelines (BM25 index) | instant |
| 8 | Define ADRE prompt builder (indication + evidence + guidelines) | instant |
| 9 | Build HuggingFace training dataset | ~2 min |
| 10 | Load Qwen2.5-3B-Instruct with QLoRA (4-bit) | ~5 min |
| 11 | **LoRA training** (3 epochs) | ~2–4 hours |
| 12 | **Run inference** on eval set with ADRE prompts | ~10–20 min |
| 13 | Compute all 7 evaluation metrics | ~5 min |
| 14 | Zip outputs for download | instant |

**Total estimated time: 3–5 hours** (most is training)

---

### Step 5 — Download the Results

After the notebook finishes, go to the **Output** tab on the right sidebar and download:

- `adre_lora_metrics.json` — the 7 evaluation metrics
- `adre_lora_predictions.jsonl` — generated reports + references

---

### Step 6 — Save Results Locally

Copy the downloaded files into your local repo:

```
adre-cxr/outputs/adre_lora_metrics.json
adre-cxr/outputs/adre_lora_predictions.jsonl
```

---

## What Makes This Different from the Old LoRA

The old notebook (02) trained like this:
```
Input:  "Indication: Positive TB test"
Output: "FINDINGS: ... IMPRESSION: ..."
```

This new notebook (03) trains like this:
```
Input:  "Indication: Positive TB test
         
         Pathology Scores:
           Consolidation: 0.71 [HIGH]
           Atelectasis:   0.65 [HIGH]
           Infiltration:  0.61 [MOD]
           ...
         
         Regional Burden:
           Upper Left:  top=Mass (0.66)
           Upper Right: top=Atelectasis (0.66)
           ...
         
         Image Features:
           Opacity density: 0.64
           CTR proxy:       0.48
         
         Relevant Guidelines:
           [Fleischner: Consolidation]
           Consolidation is a homogeneous increase...
           
           [ACR: CXR Reporting Standards]
           A complete chest radiograph report should..."

Output: "FINDINGS: ... IMPRESSION: ..."
```

The model learns to read and use evidence + guidelines — just like the ADRE agentic pipeline does in zero-shot and few-shot.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `bitsandbytes` error / no GPU detected | The notebook auto-falls back to fp16. Let it run. |
| Cell 6 (evidence extraction) is slow | Normal — it's running DenseNet on every image. Kaggle T4 takes ~20–40 min for the full dataset. |
| Cell 11 (training) runs out of memory | Reduce `per_device_train_batch_size` to 1 (already set) or reduce `max_seq_length` from 2048 to 1536 |
| `triton` not found | Run `!pip install triton` in a new cell before cell 10, then restart and re-run |
| Kaggle session times out during training | Save checkpoints are enabled every epoch. Re-run only cell 11 onwards. |

---

## After You Get the Results

Compare the new ADRE-LoRA metrics against the old experiments:

| Setting | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore |
|---|---|---|---|---|---|
| Zero-Shot (ADRE pipeline) | 0.0835 | 0.0020 | 0.0980 | 0.1137 | 0.8311 |
| Few-Shot (ADRE pipeline) | 0.1113 | 0.0085 | 0.1290 | 0.1330 | 0.8389 |
| Old LoRA (indication only) | 0.3257 | 0.1038 | 0.3456 | 0.3025 | 0.9058 |
| **ADRE-LoRA (evidence + guidelines)** | TBD | TBD | TBD | TBD | TBD |

The ADRE-LoRA should score higher than the old LoRA because the model is trained to use the full ADRE evidence context at inference time — the same context the zero-shot and few-shot runs used.

---

*Document generated: July 31, 2026*
