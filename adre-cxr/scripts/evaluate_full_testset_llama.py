#!/usr/bin/env python3
"""Evaluate Llama3-OpenBioLLM-8B Zero-Shot, Few-Shot (3-Shot), and LoRA Fine-Tuned models on the full 334-record Validation/Test Split with incremental caching."""

from __future__ import annotations

import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Monkeypatch torch.load to bypass weights_only=True issues
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

import pandas as pd
import transformers
# Bypass transformers PyTorch version restriction check for loading legacy .bin weights
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import PeftModel
from sklearn.model_selection import train_test_split

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from src.data_prep.parse_reports import parse_reports_df

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

def normalize_report(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", "*").replace("•", "*")
    return " ".join(text.split())

def compute_all_eval_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    predictions = [normalize_report(p) for p in predictions]
    references = [normalize_report(r) for r in references]

    smoothie = SmoothingFunction().method1
    refs_tokens = [[r.split()] for r in references]
    hyps_tokens = [p.split() for p in predictions]

    bleu1 = corpus_bleu(refs_tokens, hyps_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs_tokens, hyps_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)

    meteor_scores = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
    meteor = sum(meteor_scores) / len(meteor_scores)

    _, _, f1 = bert_score(list(predictions), list(references), lang="en", model_type="roberta-large", verbose=False, device="cpu")
    bertscore_f1 = float(f1.mean())

    return {
        "bleu1": float(bleu1),
        "bleu2": float(bleu2),
        "bleu3": float(bleu3),
        "bleu4": float(bleu4),
        "rouge_l": float(rouge_l),
        "meteor": float(meteor),
        "bertscore_f1": float(bertscore_f1),
    }

def load_cache(cache_path: Path) -> dict[str, str]:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_path: Path, data: dict[str, str]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="/home/tukl/Taimoor/ADRE-CXR/weights/Llama3-OpenBioLLM-8B", help="Base model ID or path")
    parser.add_argument("--stop-after-pass", type=int, default=0, help="Stop after pass 1, 2, or 3 (0 for all)")
    args = parser.parse_args()

    csv_path = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "indiana_reports.csv"
    print(f"Loading and parsing reports from {csv_path}...")
    df_raw = pd.read_csv(str(csv_path))
    reports = parse_reports_df(df_raw)
    print(f"Total valid reports: {len(reports)}")

    train_records, val_records = train_test_split(reports, test_size=0.1, random_state=42)
    print(f"Validation Split Count: {len(val_records)} records.")

    # Select 3 fixed few-shot examples from train_records
    few_shot_examples = train_records[:3]

    cache_dir = ROOT / "output"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    zero_cache_path = cache_dir / "eval_cache_zero_shot_llama.json"
    few_cache_path = cache_dir / "eval_cache_few_shot_llama.json"
    lora_cache_path = cache_dir / "eval_cache_lora_llama.json"

    zero_cache = load_cache(zero_cache_path)
    few_cache = load_cache(few_cache_path)
    lora_cache = load_cache(lora_cache_path)

    instruction = (
        "Write a structured radiology report findings and impression based on the patient's indication and comparison details. "
        "Write two sections exactly: FINDINGS: and IMPRESSION:. Rules: Follow standard radiology phrasing."
    )

    references = [f"FINDINGS:\n{item['findings']}\n\nIMPRESSION:\n{item['impression']}" for item in val_records]

    # Need model loading if any cache is incomplete
    need_base_model = (len(zero_cache) < len(val_records)) or (len(few_cache) < len(val_records) and args.stop_after_pass != 1)
    need_lora_model = (len(lora_cache) < len(val_records)) and (args.stop_after_pass not in (1, 2))

    base_model = None
    tokenizer = None

    if need_base_model or need_lora_model:
        print(f"\nLoading tokenizer for {args.model_id}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Define standard Llama-3 instruct chat template if not present
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                "{% endif %}"
            )

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading base model {args.model_id} in 4-bit precision...")
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        base_model.eval()

    # -------------------------------------------------------------
    # 1. Zero-Shot Evaluation (Base Model)
    # -------------------------------------------------------------
    print("\n==================================================")
    print(f" 1/3 ZERO-SHOT EVALUATION ({len(zero_cache)}/{len(val_records)} CACHED)")
    print("==================================================")
    
    for idx, item in enumerate(val_records, 1):
        uid_key = str(item["uid"])
        if uid_key in zero_cache:
            continue

        input_text = f"Patient Indication: {item['indication']}\nComparison: {item['comparison']}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        zero_cache[uid_key] = pred_text
        save_cache(zero_cache_path, zero_cache)

        if len(zero_cache) % 10 == 0 or len(zero_cache) == len(val_records):
            print(f"[Zero-Shot] Cached {len(zero_cache)}/{len(val_records)} records...")
            sys.stdout.flush()

    zero_shot_preds = [zero_cache[str(item["uid"])] for item in val_records]
    print(f"[SUCCESS] Zero-Shot Pass Complete: All {len(zero_shot_preds)} predictions saved.")

    if args.stop_after_pass == 1:
        print("\n[INFO] Stopped after Pass 1 as requested (--stop-after-pass 1). Progress saved to disk!")
        sys.exit(0)

    # -------------------------------------------------------------
    # 2. Few-Shot Evaluation (Base Model + 3 In-Context Examples)
    # -------------------------------------------------------------
    print("\n==================================================")
    print(f" 2/3 FEW-SHOT (3-SHOT) EVALUATION ({len(few_cache)}/{len(val_records)} CACHED)")
    print("==================================================")
    
    for idx, item in enumerate(val_records, 1):
        uid_key = str(item["uid"])
        if uid_key in few_cache:
            continue

        messages = [{"role": "system", "content": instruction}]
        for ex in few_shot_examples:
            ex_input = f"Patient Indication: {ex['indication']}\nComparison: {ex['comparison']}"
            ex_output = f"FINDINGS:\n{ex['findings']}\n\nIMPRESSION:\n{ex['impression']}"
            messages.append({"role": "user", "content": ex_input})
            messages.append({"role": "assistant", "content": ex_output})

        current_input = f"Patient Indication: {item['indication']}\nComparison: {item['comparison']}"
        messages.append({"role": "user", "content": current_input})

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        few_cache[uid_key] = pred_text
        save_cache(few_cache_path, few_cache)

        if len(few_cache) % 10 == 0 or len(few_cache) == len(val_records):
            print(f"[Few-Shot] Cached {len(few_cache)}/{len(val_records)} records...")
            sys.stdout.flush()

    few_shot_preds = [few_cache[str(item["uid"])] for item in val_records]
    print(f"[SUCCESS] Few-Shot Pass Complete: All {len(few_shot_preds)} predictions saved.")

    if args.stop_after_pass == 2:
        print("\n[INFO] Stopped after Pass 2 as requested (--stop-after-pass 2). Progress saved to disk!")
        sys.exit(0)

    # -------------------------------------------------------------
    # 3. LoRA Fine-Tuned Evaluation (Base Model + LoRA Adapters)
    # -------------------------------------------------------------
    print("\n==================================================")
    print(f" 3/3 LORA FINE-TUNED EVALUATION ({len(lora_cache)}/{len(val_records)} CACHED)")
    print("==================================================")
    
    if len(lora_cache) < len(val_records):
        adapter_path = ROOT / "weights" / "llama3-openbiollm-8b-lora"
        print(f"Attaching LoRA adapter from {adapter_path}...")
        lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        lora_model.eval()

        for idx, item in enumerate(val_records, 1):
            uid_key = str(item["uid"])
            if uid_key in lora_cache:
                continue

            input_text = f"Patient Indication: {item['indication']}\nComparison: {item['comparison']}"
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = lora_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    min_new_tokens=40,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            lora_cache[uid_key] = pred_text
            save_cache(lora_cache_path, lora_cache)

            if len(lora_cache) % 10 == 0 or len(lora_cache) == len(val_records):
                print(f"[LoRA FT] Cached {len(lora_cache)}/{len(val_records)} records...")
                sys.stdout.flush()

    lora_preds = [lora_cache[str(item["uid"])] for item in val_records]
    print(f"[SUCCESS] LoRA FT Pass Complete: All {len(lora_preds)} predictions saved.")

    # -------------------------------------------------------------
    # Compute and Print Final Metrics Summary Table
    # -------------------------------------------------------------
    print("\nComputing corpus metrics for all 3 evaluation modes...")
    zero_metrics = compute_all_eval_metrics(zero_shot_preds, references)
    few_metrics = compute_all_eval_metrics(few_shot_preds, references)
    lora_metrics = compute_all_eval_metrics(lora_preds, references)

    print("\n" + "=" * 65)
    print("FINAL EVALUATION METRICS ON FULL VALIDATION SPLIT (334 RECORDS)")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Zero-Shot':<12} | {'Few-Shot':<12} | {'LoRA FT':<12}")
    print("-" * 65)
    metrics_keys = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor", "bertscore_f1"]
    for k in metrics_keys:
        print(f"{k:<15} | {zero_metrics[k]:<12.4f} | {few_metrics[k]:<12.4f} | {lora_metrics[k]:<12.4f}")
    print("=" * 65)

    summary_file = cache_dir / "full_val_eval_metrics_llama.json"
    results_data = {
        "num_records": len(val_records),
        "metrics": {
            "zero_shot": zero_metrics,
            "few_shot": few_metrics,
            "lora_ft": lora_metrics
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nDetailed metrics saved to: {summary_file}")

if __name__ == "__main__":
    main()
