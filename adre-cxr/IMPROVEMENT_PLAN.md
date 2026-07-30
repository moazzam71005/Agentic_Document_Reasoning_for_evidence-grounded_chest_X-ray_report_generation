# Further Improvement Plan — ADRE-CXR

> Concrete, prioritized steps to improve report generation quality beyond the current results.
> Organized by effort level: Quick Wins → Medium Effort → Research-Level.

---

## Current Best Results (Baseline to Beat)

| Metric | Current Best | Configuration |
|---|---|---|
| BLEU-1 | 0.3257 | Qwen-3B, 3ep, full LoRA, nucleus sampling |
| BLEU-4 | 0.1050 | Llama-3-8B, 6ep, full LoRA, beam search + truncation |
| ROUGE-L | 0.3456 | Qwen-3B, 3ep, full LoRA, nucleus sampling |
| METEOR | 0.3343 | Llama-3-8B, 6ep, full LoRA, beam search + truncation |
| BERTScore F1 | 0.9058 | Qwen-3B, 3ep, full LoRA, nucleus sampling |

---

## Priority 1 — Quick Wins (1–3 days, no new training required)

### 1.1 Ensemble / Hybrid Decoding

**What:** Run inference with both Qwen-3B (nucleus sampling) and Llama-3-8B (beam search), then pick the best output per study using BERTScore as a selector.

**Why:** Qwen wins on BLEU-1 and BERTScore; Llama wins on BLEU-4 and METEOR. A per-sample selector could get the best of both.

**How:**
```python
# Generate both outputs
qwen_out = generate(model=qwen, prompt=indication)
llama_out = generate(model=llama, prompt=indication)

# Score against a self-reference (BERTScore between the two)
# Pick the output with higher BERTScore vs the other
```

**Expected gain:** +2–5% on most metrics at zero extra training cost.

---

### 1.2 Longer Output Control

**What:** Current `min_length=40` may be too short. Set `min_length=80–100` to encourage complete Findings + Impression sections.

**Why:** BLEU and ROUGE penalize outputs that are too short relative to reference. IU X-Ray reference reports are typically 50–150 tokens.

**How:** Change `min_new_tokens=80` in the inference config.

---

### 1.3 Length Penalty Tuning

**What:** Try `length_penalty=0.8` (slight negative bias against very long sequences) with beam search to control verbosity.

**Why:** Overly long generations introduce irrelevant tokens that dilute BLEU precision.

---

### 1.4 Better Post-Processing

**What:** Extend `clean_predictions()` to also:
- Remove any text before "Findings:" or "FINDINGS"
- Strip repeated sentences (deduplicate consecutive identical sentences)
- Normalize whitespace and punctuation

**Why:** Formatting artifacts in the output hurt all n-gram metrics even when content is correct.

---

## Priority 2 — Medium Effort (1–2 weeks)

### 2.1 Train on Larger Dataset — MIMIC-CXR

**What:** Fine-tune the LoRA model on MIMIC-CXR (377k reports) instead of or in addition to IU X-Ray (~3,900 reports).

**Why:** IU X-Ray is a small dataset (~3,000 training samples after split). MIMIC-CXR is 100x larger and would provide a much richer training signal.

**Prerequisites:**
- PhysioNet account + CITI training (free, ~2 hours)
- Access approval for MIMIC-CXR (usually approved within days)

**Expected gain:** BLEU-4 could reach 0.15–0.20 range, matching published SOTA.

**Kaggle note:** MIMIC-CXR is available on PhysioNet but can be downloaded to Kaggle via the `physionet` CLI.

---

### 2.2 Increase LoRA Rank

**What:** Try r=64 or r=128 with proportionally adjusted `alpha` (alpha = 2×rank is a common rule of thumb).

**Why:** Higher rank = more trainable parameters = more expressive adaptation. Our current best used r=16–32. r=64 may close the gap on METEOR and BLEU-4.

**Compute note:** r=64 on 8B model on a single T4/A100 is feasible within Kaggle's 30-hour/week limit.

---

### 2.3 Instruction Tuning Format Refinement

**What:** Restructure the training prompt to explicitly ask for structured output:

```
### Instruction:
Generate a radiology report for the following chest X-ray.

### Indication:
{indication}

### Report:
**FINDINGS:** {findings}

**IMPRESSION:** {impression}
```

**Why:** Current training format may mix findings and impression in one block. Explicitly separating them helps the model learn section boundaries, reducing the need for `clean_predictions()`.

---

### 2.4 Retrieval-Augmented Generation (RAG) for Fine-Tuned Model

**What:** At inference time, retrieve similar past reports from the training set using BM25/FAISS and inject them into the prompt (like dynamic few-shot, but from the training data).

**Why:** Static 3-shot examples are fixed; dynamic retrieval gives examples that are semantically similar to the input indication.

**How:**
```python
# Build BM25 index over training indications
similar = retrieve_top_k(query=indication, index=train_index, k=3)
prompt = few_shot_prompt(examples=similar) + indication
```

**Expected gain:** +5–10% on BLEU and ROUGE vs static few-shot.

---

### 2.5 Separate Section Generation

**What:** Train two separate LoRA adapters (or two LoRA prompt prefixes):
1. One for generating **Findings** from the indication
2. One for generating **Impression** from the Findings

**Why:** Findings and Impression are stylistically different — Findings is a detailed enumeration; Impression is a brief summary/conclusion. A single model may compromise between the two styles.

---

## Priority 3 — Research-Level (2–4 weeks)

### 3.1 Add a CXR Vision Encoder

**What:** Replace the TorchXRayVision DenseNet feature extraction with a proper vision-language model:
- **BioViL-T** (Microsoft, open source) — CXR-specific vision encoder with text alignment
- **CheXNet** (Stanford) — Chest X-ray 14-label classifier
- **LLaVA-Med** — General medical VLM

**Why:** Our current pipeline uses extracted features (pathology probabilities, opacity scores) but does not pass the raw image to the LLM. Adding a proper visual encoder allows the LLM to "see" the X-ray directly.

**Architecture change:**
```
[CXR Image] → [Vision Encoder] → [Image Embeddings]
                                        ↓
[Indication] → [Text Encoder] → [Text Embeddings]
                                        ↓
                              [LLM: Findings + Impression]
```

**Expected gain:** BLEU-4 could reach 0.15–0.25, ROUGE-L 0.40+, matching top-tier CXR systems.

---

### 3.2 RLHF / Reward Model from Radiologist Feedback

**What:** Collect a small set (~200 samples) of human expert ratings on generated reports. Train a reward model. Use PPO or GRPO to fine-tune the LLM toward higher-rated outputs.

**Why:** BLEU/ROUGE/METEOR are imperfect proxies for clinical quality. A reward model trained on human judgments would directly optimize for what radiologists care about (completeness, clinical accuracy, fluency).

**Feasibility note:** Can use LLM-as-judge (GPT-4 or a medical LLM) as a proxy reward model if human radiologist access is unavailable.

---

### 3.3 Contrastive Decoding

**What:** Use an amateur model (small, untrained) alongside the expert model during decoding. Amplify tokens where the expert is more confident than the amateur.

**Why:** Reduces generic outputs and boosts domain-specific vocabulary in the generated reports.

---

### 3.4 Multi-Image Input

**What:** Most IU X-Ray studies have both a frontal (PA) and a lateral view. Currently we use only one image. Feed both views to the model.

**Why:** Radiologists always read both views together. Lateral view provides complementary information (retrocardiac space, posterior costophrenic angles, spinal alignment).

---

## Recommended Next Steps (in order)

| Step | Effort | Expected Impact | Try First? |
|---|---|---|---|
| Better post-processing + longer min_length | 1 day | +2–4% BLEU | ✅ Yes |
| Qwen + Llama ensemble with BERTScore selector | 1 day | +2–5% all metrics | ✅ Yes |
| Dynamic RAG few-shot (retrieve similar reports) | 3 days | +5–10% | ✅ Yes |
| Higher LoRA rank (r=64) | 2 days training | +3–7% BLEU-4 | Yes |
| Better instruction format (separate sections) | 2 days | +3–5% | Yes |
| MIMIC-CXR fine-tuning | 1–2 weeks | +15–30% | High priority |
| CXR vision encoder integration | 2–4 weeks | +20–40% | Medium priority |
| RLHF with LLM-as-judge | 2–4 weeks | Variable | Research priority |

---

## Target Scores to Aim For

Based on published literature and the trajectory of our experiments:

| Metric | Current Best | Short-term Target (1–2 weeks) | Long-term Target (1–2 months) |
|---|---|---|---|
| BLEU-1 | 0.3257 | 0.36–0.40 | 0.42–0.48 |
| BLEU-4 | 0.1050 | 0.13–0.16 | 0.16–0.20 |
| ROUGE-L | 0.3456 | 0.38–0.42 | 0.40–0.45 |
| METEOR | 0.3343 | 0.35–0.38 | 0.40–0.45 |
| BERTScore F1 | 0.9058 | 0.91–0.92 | 0.92–0.94 |

---

## Summary

The biggest remaining gaps are:
1. **No visual input to the LLM** — the most impactful single change
2. **Training data scale** — MIMIC-CXR would be transformative
3. **Output quality control** — post-processing and section-aware generation can improve metrics without retraining

The current results (BLEU-4 ~0.10, BERTScore ~0.90) are strong given the pipeline constraints and demonstrate that the ADRE adaptation works. The path to competitive SOTA runs through a proper vision encoder + larger training data.

---

*Document generated: July 30, 2026*
