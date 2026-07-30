# Presentation Guide — ADRE-CXR Experiments

> Slide-by-slide breakdown of what to say and show when presenting to your supervisor.
> Estimated total: **12–14 slides**, ~15–20 minutes.

---

## Slide 1 — Title Slide

**Title:** Applying ADRE to Chest X-Ray Report Generation
**Subtitle:** Evaluation Across Zero-Shot, Few-Shot, and LoRA Fine-Tuning
**Authors:** Your name + partner's name
**Date:** July 2026

---

## Slide 2 — Motivation (Why this problem?)

**What to say:**
- Radiologists write hundreds of reports per day — tedious and error-prone
- Automatic radiology report generation can assist, speed up, and standardize reporting
- ADRE paper proposed a structured, evidence-grounded framework — we applied it to chest X-rays

**What to show:**
- 1–2 example chest X-ray images alongside a sample report (Findings + Impression)
- One sentence: "Can an AI system look at an X-ray and generate a report a radiologist would write?"

---

## Slide 3 — ADRE Framework Overview

**What to say:**
- ADRE = Agentic Document Reasoning for Evidence-Grounded report generation
- Original paper was for EEG reports — we adapted it to CXR
- Key idea: instead of directly prompting an LLM, build a structured evidence packet first

**What to show:**
A 4-step pipeline diagram:
```
[X-ray Image]
     ↓
[Evidence Extraction]  ← DenseNet121 classifier, image features, quality checks
     ↓
[Knowledge Retrieval]  ← BM25 + FAISS over Fleischner / ACR guidelines
     ↓
[Agentic Reasoning Loop]  ← Gap analysis, multi-view tool planning
     ↓
[Report Generation]  ← LLM generates Findings + Impression
```

**Key message:** The pipeline is modular — evidence → knowledge → reasoning → generation.

---

## Slide 4 — Dataset

**What to say:**
- IU X-Ray (Indiana University, Open-I) — public domain, no credentialing required
- ~3,955 reports, ~7,470 images
- Reports already have Findings + Impression sections — perfect for evaluation
- We used a 26-study dev set for initial runs; partner expanded to 334-record validation set

**What to show:**
| Property | Value |
|---|---|
| Source | Kaggle — `raddar/chest-xrays-indiana-university` |
| Reports | ~3,955 |
| Our dev set | 26 studies |
| Partner's val set | 334 studies |

---

## Slide 5 — Three Experimental Conditions

**What to say:**
We evaluated three ways to use the LLM in the pipeline:

1. **Zero-Shot** — No examples, no fine-tuning. LLM gets evidence + guidelines and generates cold.
2. **Few-Shot (3-Shot)** — Three example reports are prepended to the prompt at inference. No training.
3. **LoRA Fine-Tuning** — The LLM is trained on IU X-Ray report pairs before inference.

**What to show:**
A simple 3-column diagram or table:

| | Zero-Shot | Few-Shot | LoRA |
|---|---|---|---|
| Training required? | No | No | Yes |
| Examples in prompt? | No | 3 | No |
| Model updated? | No | No | Yes (adapters) |
| GPU needed? | No | No | Yes (Kaggle T4 / RTX 3080) |

---

## Slide 6 — Evaluation Metrics

**What to say:**
We use 7 standard metrics from the radiology NLP literature:
- **BLEU-1/2/3/4**: Measures exact n-gram overlap between generated and reference report
- **ROUGE-L**: Measures longest common subsequence match
- **METEOR**: Like BLEU but also accounts for synonyms and paraphrases
- **BERTScore F1**: Semantic similarity using a pre-trained language model (RoBERTa-large) — best for clinical text where paraphrasing is common

**Key note for supervisor:** BERTScore is the most meaningful metric here because clinical language allows many valid phrasings for the same finding.

---

## Slide 7 — Our Results (26 Studies)

**What to say:**
Starting with our initial 26-study runs using Qwen2.5-1.5B LoRA and Groq (Llama-3-70B) for zero/few-shot:

**What to show:**

| Setting | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore F1 |
|---|---|---|---|---|---|
| Zero-Shot | 0.0835 | 0.0020 | 0.0980 | 0.1137 | 0.8311 |
| Few-Shot (3-shot) | 0.1113 | 0.0085 | 0.1290 | 0.1330 | 0.8389 |
| LoRA (Qwen2.5-1.5B) | **0.2443** | **0.0955** | **0.3221** | **0.2647** | **0.8863** |

**Key takeaway:** LoRA fine-tuning gives a massive improvement. Even a 1.5B model fine-tuned on in-domain data outperforms a 70B zero-shot model.

---

## Slide 8 — Partner's Optimized Results (334 Studies)

**What to say:**
The partner scaled up: larger models (Qwen-3B, Llama-3-8B), more training, and systematic hyperparameter tuning on the full 334-record validation set.

**What to show:**

| Setting | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore F1 |
|---|---|---|---|---|---|
| Zero-Shot (Qwen-3B) | 0.1109 | 0.0077 | 0.1375 | 0.1775 | 0.8539 |
| Few-Shot (Qwen-3B) | 0.1926 | 0.0354 | 0.2235 | 0.2515 | 0.8804 |
| LoRA Qwen-3B (3ep, best) | **0.3257** | 0.1038 | **0.3456** | 0.3025 | **0.9058** |
| LoRA Llama-3-8B (6ep, best) | 0.2769 | **0.1050** | 0.3253 | **0.3343** | 0.8837 |

**Key takeaway:** Best BLEU-4 = 0.1050 (Llama-3-8B). Best BERTScore = 0.9058 (Qwen-3B). Both beat the initial 1.5B LoRA significantly.

---

## Slide 9 — Key Optimization Findings

**What to say:**
Three concrete engineering insights came out of the partner's experiments:

**What to show — 3 boxes or bullet points:**

**1. LoRA Coverage Matters**
- Full 7-projection coverage (q,k,v,o,gate,up,down) vs. attention-only (q,k,v,o)
- Full coverage BLEU-4: **0.1050** vs attention-only: 0.0426 (+147%)
- MLP weights are critical for adapting report style, not just attention

**2. Post-Processing Is Essential for Llama**
- Llama-3 instruction models emit trailing dialogue tokens (`assistant:`, `user:`)
- `clean_predictions()` truncates these → BLEU-4 doubled: 0.0509 → **0.1050**

**3. Beam Search vs. Nucleus Sampling**
- 4-bit quantized models under nucleus sampling fall into repetitive loops
- Beam search (`num_beams=4`, `no_repeat_ngram_size=3`) solves this for Llama
- Qwen performs well with nucleus sampling (temp=0.1, top_p=0.9)

---

## Slide 10 — Progress Over Iterations (Summary Chart)

**What to show:** A progression bar chart or table showing how BLEU-4 improved across our iterations:

| Stage | BLEU-4 | What changed |
|---|---|---|
| Our Zero-Shot | 0.0020 | Baseline |
| Our Few-Shot | 0.0085 | +3 prompt examples |
| Our LoRA (1.5B) | 0.0955 | Fine-tuned small model |
| Partner LoRA (3B, V1) | 0.1038 | Larger model, 3 epochs |
| Partner LoRA (8B, V3 final) | **0.1050** | Larger model, 6ep, full coverage, beam search |

**Key message:** Every design decision — prompt engineering → fine-tuning → post-processing — contributed measurable gains.

---

## Slide 11 — Comparison with Published Work

**What to say:**
Our results are competitive with published CXR report generation systems, despite using a much simpler pipeline.

**What to show:**

| System | BLEU-4 | ROUGE-L | METEOR | Notes |
|---|---|---|---|---|
| R2Gen (Chen et al., 2020) | 0.165 | 0.371 | 0.187 | Dedicated vision encoder |
| RGRG (Tanida et al., 2023) | 0.168 | 0.390 | 0.196 | Region-grounded generation |
| **Our best (Llama-3-8B LoRA)** | **0.1050** | 0.3253 | **0.3343** | LLM fine-tuning, no vision encoder |
| **Partner best (Qwen-3B LoRA)** | 0.1038 | **0.3456** | 0.3025 | LLM fine-tuning, ROUGE competitive |

**Note to say:** Published systems have dedicated CXR image encoders trained on millions of images. We rely only on a text-based indication and fine-tuning — yet our METEOR and ROUGE-L are competitive.

---

## Slide 12 — What We Proved (Takeaways)

**What to say — 4 key points:**

1. **ADRE framework adapts to CXR.** The evidence → retrieval → reasoning → generation pipeline works, and each stage is replaceable or upgradeable independently.

2. **Fine-tuning beats prompting on in-domain data.** A 1.5B model fine-tuned on IU X-Ray reports outperforms a 70B zero-shot model on every metric.

3. **Engineering choices (coverage, truncation, decoding) matter as much as model scale.** A 3B model with right config beats an 8B model with wrong config on most metrics.

4. **BERTScore stays high across all conditions (0.83–0.91).** The semantic content of generated reports is reasonable even in zero-shot; fine-tuning improves wording precision.

---

## Slide 13 — Limitations

**What to say:**
Be honest about limitations — supervisors appreciate this:

- **No image encoder**: we use indication text + extracted features, not raw CXR image embeddings. A dedicated vision encoder (like BioViL or CheXAgent) could substantially help.
- **Small evaluation set for initial runs**: 26 studies is limited; partner's 334-study eval is more reliable.
- **No radiologist evaluation**: all metrics are automated. Human expert evaluation would be needed for clinical validation.
- **IU X-Ray is a relatively clean dataset**: real-world CXR reports are noisier.

---

## Slide 14 — Future Work (Optional / if time allows)

Briefly mention (details in the improvement plan):
- Add a CXR vision encoder (BioViL-T, CheXNet)
- Train on MIMIC-CXR for scale
- Physician-in-the-loop feedback / RLHF
- Structured output with anatomical region grounding

---

## Tips for the Presentation

- **Lead with results**, then explain methods. Supervisors want to see the numbers first.
- **Emphasize the progression**: zero-shot → few-shot → LoRA is a clean story.
- **Highlight the partner's contribution** separately from your baseline — show it as a systematic ablation study on top of the initial work.
- **Keep the demo ready**: have one sample report (generated vs ground truth) to show side by side.
- **Anticipate question**: "Why not just use GPT-4V?" — answer: we wanted a trainable, open-source, reproducible pipeline.

---

*Document generated: July 30, 2026*
