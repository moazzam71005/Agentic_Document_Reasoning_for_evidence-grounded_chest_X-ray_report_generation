# Implementation Plan: ADRE-CXR
## Agentic Document Reasoning for Evidence-Grounded Chest X-Ray Report Generation

**Goal:** Replicate the ADRE framework (originally built for EEG reports) on a chest
X-ray dataset, keeping the same architectural philosophy: explicit intermediate
findings representation → layout-aware guideline retrieval → agentic gap-analysis
loop → dual-stage (structured findings → narrative) report generation.

**Reference paper:** "Agentic Document Reasoning for Evidence-Grounded Clinical
Report Generation" (ADRE), Masood et al.

**Target dataset:** IU X-Ray / Open-I (Indiana University Chest X-Ray Collection)
- ~7,470 frontal/lateral chest radiographs, ~3,955 reports
- Each report has: Comparison, Indication, Findings, Impression
- Public domain, no credentialing required
- Sources (pick one, verify current availability before starting):
  - Kaggle: `raddar/chest-xrays-indiana-university`
  - HuggingFace: search `iu_xray` / `IU-Xray` datasets
  - Original: NLM Open-I API (https://openi.nlm.nih.gov/)

---

## 0. Concept mapping (ADRE-EEG → ADRE-CXR)

| ADRE (EEG) | ADRE-CXR (X-ray) |
|---|---|
| EDF signal | DICOM/PNG/JPG chest X-ray image (frontal ± lateral) |
| View Builder (montage/reference/filter/window) | Image view builder: original, lung-zone crops (upper/mid/lower × left/right), contrast/CLAHE-enhanced, frontal vs. lateral |
| SCNet → Global Abnormality Score | Pretrained CXR classifier (e.g. TorchXRayVision DenseNet) → global pathology probabilities |
| Regional Morphology Summary (region × event burden) | Per-lung-zone finding summary via lung segmentation + zone-wise classifier/Grad-CAM |
| Signal features (δ/θ power, PDR, amplitude, etc.) | Image-derived features: cardiothoracic ratio, opacity density per zone, symmetry index L/R |
| Quality indicators (power-line ratio, EMG, bad channel) | Quality indicators: rotation, inspiration adequacy, penetration/exposure, artifact/foreign body flags |
| Clinical guideline corpus (Jadeja, ANSA, ACNS…) | Radiology guideline corpus (Fleischner glossary, ACR reporting guides) |
| Findings + Impression report | Findings + Impression report (native to IU X-Ray — no reformatting needed) |
| Patient metadata (age, gender) | Patient metadata (age, gender, indication/clinical history if present) |

Everything downstream (layout-aware retrieval, agentic gap-analysis loop,
dual-stage generation, evaluation metrics, ablation design) carries over almost
unchanged — only the **evidence acquisition layer** needs to be rebuilt for images
instead of EEG signals.

---

## 1. Repository structure

```
adre-cxr/
├── data/
│   ├── raw/                     # downloaded images + reports (gitignored)
│   ├── processed/                # parsed reports, splits, cached features
│   └── guidelines/               # PDF guideline corpus
├── src/
│   ├── evidence/
│   │   ├── view_builder.py       # image view generation (crops, zones, enhancement)
│   │   ├── neural_inference.py   # pretrained classifier wrapper (global + regional)
│   │   ├── image_features.py     # CTR, opacity density, symmetry index
│   │   ├── quality_indicators.py # rotation/exposure/inspiration checks
│   │   └── evidence_packet.py    # provenance-tagged evidence packet (Φ vector)
│   ├── knowledge_index/
│   │   ├── parse_guidelines.py   # layout-aware PDF parsing → hierarchical chunks
│   │   ├── build_index.py        # BM25 + FAISS index construction
│   │   └── retrieve.py           # hybrid retrieval + context expansion
│   ├── agent/
│   │   ├── state.py              # EvidenceState (St) class, append-only updates
│   │   ├── checklist.py          # report requirement checklist (Findings/Impression)
│   │   ├── gap_analysis.py       # Δt = GapAnalysis(St, C, G(qt))
│   │   ├── planner.py            # tool planning / view+window requests
│   │   ├── consistency.py        # cross-view consistency checks
│   │   └── controller.py         # main iterative loop (Algorithm 1 equivalent)
│   ├── generation/
│   │   ├── stage_a_findings.py   # schema-constrained structured findings (T=0)
│   │   ├── stage_b_narrative.py  # narrative generation (T=0.25)
│   │   └── repair.py             # format validation + repair prompt
│   ├── llm/
│   │   └── ollama_client.py      # unified LLM call wrapper (Qwen3, Gemma3, DeepSeek-R1)
│   ├── eval/
│   │   ├── metrics.py            # BLEU, ROUGE-L, METEOR, BERTScore
│   │   ├── significance.py       # Wilcoxon, paired t-test, bootstrap
│   │   └── ablation_runner.py    # runs Table-1-style variant matrix
│   └── data_prep/
│       ├── download.py
│       ├── parse_reports.py      # split IU X-Ray reports into Findings/Impression
│       └── make_splits.py        # dev/test split (frozen hyperparams on dev only)
├── configs/
│   └── default.yaml              # α, top-k, Tmax, backbone list, paths
├── scripts/
│   ├── run_pipeline.py           # single-record end-to-end run
│   ├── run_ablation.py           # full Table 1 / Table 4 style sweep
│   └── evaluate.py
├── notebooks/                    # exploratory only, not part of pipeline
├── requirements.txt
└── README.md
```

---

## 2. Phase-by-phase implementation

### Phase 1 — Data acquisition & preparation
- [ ] Download IU X-Ray images + reports (Kaggle/HuggingFace mirror).
- [ ] Parse each report into structured fields: `comparison`, `indication`,
      `findings`, `impression` (regex/section-header split — IU X-Ray XML/CSV
      already has these tags in most mirrors; verify format for the chosen source).
- [ ] Filter to studies that have both a usable image and a non-empty
      Findings + Impression pair.
- [ ] Fix a **dev set** (~100 studies) for hyperparameter tuning and a
      **held-out test set** (~remaining, e.g. 300–400 studies) — freeze all
      hyperparameters after dev tuning, exactly as in ADRE §4.1.
- [ ] Store patient metadata (age, gender, indication) alongside each record.

### Phase 2 — Multi-modal evidence acquisition (replaces §3.2)
- [ ] **View Builder**: implement callable image transforms — original image,
      CLAHE/contrast-enhanced, lung-zone crops (6 zones: upper/mid/lower ×
      left/right via lung segmentation mask), frontal vs. lateral (if both
      available). Tag every output with a `view_id`.
- [ ] **Neural Inference**:
  - Global Abnormality Score: use a pretrained model such as
    `torchxrayvision` (DenseNet121, trained on multiple public CXR datasets) —
    returns per-pathology probabilities (14–18 labels); aggregate into a single
    `P_ab` + confidence, analogous to SCNet's Pab.
  - Regional Morphology Summary: run a lung segmentation model (e.g. U-Net
    lung mask, several pretrained checkpoints are open source) to get zone
    masks, then compute per-zone abnormality burden (Grad-CAM heatmap mass
    per zone, or per-zone classifier score) — this is the direct analogue of
    Equation 1 (region × event-type burden).
- [ ] **Image Features & Quality Indicators**: implement cardiothoracic ratio,
      per-zone opacity density, L/R symmetry index, rotation estimate (clavicle
      symmetry), exposure/penetration heuristic, inspiration adequacy
      (rib count), foreign body/artifact flag.
- [ ] **Evidence Packet**: assemble into a single packet with provenance vector
      `Φ = (view_id, zone, params, tool)`, mirroring ADRE's Φ = (view_id,
      channels, t_start, t_end, params, tool).

### Phase 3 — Layout-aware knowledge indexing (replaces §3.3, unchanged in spirit)
- [ ] Collect guideline corpus PDFs (Fleischner glossary, ACR reporting
      guidance, any open CXR reporting standard docs). Store under
      `data/guidelines/`.
- [ ] Parse with a layout-aware PDF parser (MinerU, as in the original paper,
      or an open alternative) into hierarchical chunks:
      `{id, parent_id, title_path, page_span, sibling_idx}`.
- [ ] Build hybrid retrieval index: BM25Okapi (k1=1.5, b=0.75) + FAISS dense
      index (all-MiniLM-L6-v2, d=384), fused with α=0.5 (tune on dev set only).
- [ ] Implement context expansion: pull ±1 structural siblings under the same
      `parent_id`, prepend `title_path` as scope anchor, cap at 500 words.

### Phase 4 — Agentic reasoning engine (replaces §3.4, structurally identical)
- [ ] `EvidenceState (St)`: append-only store of image evidence (per view_id),
      quality flags, patient metadata, retrieved guideline passages.
- [ ] `Checklist (C)`: required report components — Findings (per-zone /
      per-organ description) and Impression (summary diagnosis) — derived from
      the IU X-Ray report template.
- [ ] `Gap analysis`: Δt = GapAnalysis(St, C, G(qt)) — identify missing zone
      coverage, weak-confidence findings, or view-inconsistent predictions.
- [ ] `Tool planning`: request additional views (e.g. re-crop a zone, apply
      contrast enhancement, check lateral view) when gaps are found.
- [ ] `Consistency checks`: flag conflicts (e.g. high global abnormality score
      but no zone shows elevated burden) → trigger re-analysis rather than
      forcing a conclusion.
- [ ] `Controller`: implement the main loop (Algorithm 1 equivalent) with a
      step budget `Tmax` (tune on dev, e.g. start at 10–15 given images need
      fewer views than multi-montage EEG) and a cap on distinct views/zooms
      requested per study.

### Phase 5 — Dual-stage report generation (replaces §3.5)
- [ ] **Stage A** (T=0.0): schema-constrained LLM call producing a structured
      findings JSON — separate `observations`, `localization` (per zone),
      `qualifiers` (from quality indicators / uncertainty).
- [ ] **Stage B** (T=0.25, top-p sampling): narrative generation from Stage-A
      JSON + retrieved report-writing guideline passages → produces the final
      Findings + Impression text, matching IU X-Ray's report format for
      direct comparison against ground truth.
- [ ] **Format validation & repair**: if Stage A JSON is malformed or Stage B
      output is missing required sections, issue a repair prompt using the
      same grounded inputs.

### Phase 6 — LLM backbones
- [ ] Run everything locally via Ollama (matches ADRE's reproducibility setup).
- [ ] Evaluate at least 2–3 backbones for comparability with the original
      paper's design, e.g. `qwen3:8b`, `gemma3:12b`, `deepseek-r1:8b` (use
      whatever is locally feasible on your GPU/VRAM).

### Phase 7 — Experimental design (mirrors §4.3 Table 1 + §4.4 Table 4)
- [ ] Implement the same 6-way ablation matrix:
      Baseline / Agentic-only / RAG-only / Agentic+Dual / RAG+Agentic /
      Full ADRE-CXR — each run across all backbones.
- [ ] Implement the retrieval ablation: no-retrieval, sparse-only, dense-only,
      flat-hybrid, layout-aware hybrid — fixed on the best agentic+dual config.
- [ ] Freeze all hyperparameters (α, top-k, Tmax, view/zoom limits) after dev
      tuning; only tune on the ~100-study dev split.

### Phase 8 — Evaluation
- [ ] Metrics: BLEU-4, ROUGE-L, METEOR, BERTScore (report RoBERTa-large as
      primary + DistilBERT/DeBERTa for robustness, exactly as ADRE Table 2/3).
- [ ] Significance testing: Wilcoxon signed-rank, paired t-test, bootstrap
      (10,000 resamples) on per-record BERTScore deltas vs. baseline.
- [ ] Reproduce ADRE-style tables:
      - Table 2 analogue: METEOR + BERTScore per backbone per variant
      - Table 3 analogue: cross-embedding-backbone robustness
      - Table 4 analogue: retrieval-stack ablation
      - Figure 3/4 analogues: stepwise ablation trend, dual-stage vs.
        single-stage, layout-aware vs. flat retrieval
- [ ] Qualitative case table (Table 5 analogue): 1 success case + 2–3 failure
      modes specific to X-ray (e.g. missing lateral view → localization
      ambiguity; low-exposure image → false "normal" call; overlapping
      pathologies → impression drift).

### Phase 9 — Write-up
- [ ] Draft results section following the same structure as the ADRE paper
      (5.1–5.4), substituting EEG-specific language for CXR-specific language.
- [ ] Note dataset-specific limitations (e.g. IU X-Ray reports are shorter/less
      standardized than the EEG references used in ADRE; single-institution
      dataset; frontal-only studies for many records).

---

## 3. Key implementation notes / gotchas for the coding agent

- **No fine-tuning** — this is a zero-shot, inference-only pipeline (like
  ADRE). All "learning" happens via retrieval + prompting + pretrained
  off-the-shelf classifiers (TorchXRayVision or similar), not by training a
  new model on IU X-Ray.
- **Provenance tagging is mandatory** — every piece of evidence (crop, zone
  score, feature) must carry a `view_id`/`Φ` so generated claims can be
  traced back, this is central to the paper's "evidence-grounded" claim.
- **Append-only state** — never overwrite `St`; new evidence is added
  alongside old evidence from different views, exactly as in ADRE §3.4.
- **Dev/test separation is strict** — do not tune α, top-k, Tmax, or view
  budgets on the test set; freeze after dev tuning only, then run once.
- **Use existing pretrained CXR models** rather than training classifiers
  from scratch — `torchxrayvision` is the fastest path to a working
  Global Abnormality Score + pathology probabilities.
- **Report structure reuse** — since IU X-Ray reports already have
  Findings/Impression, the checklist `C` and Stage-A schema can map almost
  directly onto the dataset's native fields, minimizing reformatting work.
- **Guideline corpus is the main manual step** — sourcing 2–4 open PDFs
  (Fleischner glossary, ACR guidance) needs to happen early since retrieval
  quality depends on it.

---

## 4. Suggested milestone order (for standup/reporting to supervisor)

1. Data pipeline + parsed Findings/Impression pairs, dev/test split frozen.
2. Evidence acquisition layer producing a working evidence packet for one image.
3. Guideline corpus parsed + hybrid retrieval index built and query-tested.
4. Baseline variant (no agentic loop, no RAG, no dual-stage) running
   end-to-end on dev set — first BERTScore numbers.
5. Full agentic loop + dual-stage generation working end-to-end.
6. Full 6-way ablation matrix run on test set + significance testing.
7. Retrieval-stack ablation (Table 4 analogue).
8. Qualitative case analysis + write-up.
