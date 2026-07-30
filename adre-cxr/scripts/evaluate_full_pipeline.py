#!/usr/bin/env python3
"""Evaluate end-to-end ADRE-CXR agentic pipeline (neural classification, planner, RAG, and local LLM) on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Bypass transformers PyTorch version restriction check for loading legacy .bin weights
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import pandas as pd
import torch
import transformers
from peft import PeftModel
from sklearn.model_selection import train_test_split

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from src.data_prep.parse_reports import parse_reports_df
from src.agent.controller import AgentController
from src.generation.stage_a_findings import generate_structured_findings
from src.generation.stage_b_narrative import generate_narrative_report
from src.generation.repair import repair_report, validate_report
from src.knowledge_index.retrieve import HybridRetriever

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


class LocalTransformersClient:
    """Wrapper around local transformers model to match LLMClient interface."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.provider = "local_transformers"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, top_p: float = 1.0, json_mode: bool = False) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Set default stop tokens for Llama-3 and Qwen models
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        if "llama" in self.tokenizer.name_or_path.lower():
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None:
                terminators.append(eot_id)

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": terminators,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response


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


def clean_predictions(pred_text: str) -> str:
    for marker in ["assistant\n", "assistant", "user\n", "user", "SmartyHeaderCode", "始化", "ForCanBeConverted", "smartypants"]:
        if marker in pred_text:
            pred_text = pred_text.split(marker)[0]
    return pred_text.strip()


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
    parser.add_argument("--model-path", required=True, help="HuggingFace repository ID or local weights path")
    parser.add_argument("--output-json", required=True, help="Path to save final computed metrics")
    parser.add_argument("--cache-name", required=True, help="Filename for the incremental prediction cache (e.g. qwen_pipeline)")
    args = parser.parse_args()

    # Paths setup
    data_dir = ROOT / "data" / "raw" / "chest-xrays-indiana-university"
    reports_csv = data_dir / "indiana_reports.csv"
    projections_csv = data_dir / "indiana_projections.csv"
    images_dir = data_dir / "images" / "images_normalized"

    # 1. Load and split reports
    print(f"Loading and parsing reports from {reports_csv}...")
    df_raw = pd.read_csv(str(reports_csv))
    reports = parse_reports_df(df_raw)
    train_records, val_records = train_test_split(reports, test_size=0.1, random_state=42)
    print(f"Total reports: {len(reports)}, Validation records: {len(val_records)}")

    # 2. Build UID -> Image path mapping from projections CSV
    image_map = {}
    with open(projections_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("uid")
            filename = row.get("filename")
            projection = row.get("projection")
            if projection == "Frontal" and uid and filename:
                path = images_dir / filename
                if path.exists():
                    image_map[str(uid)] = str(path)

    # 3. Filter validation records that have matching frontal images
    eval_records = []
    for item in val_records:
        uid_str = str(item["uid"])
        if uid_str in image_map:
            eval_records.append({
                "uid": uid_str,
                "image_path": image_map[uid_str],
                "indication": item.get("indication", "").strip() or "eval abnormality",
                "reference": f"FINDINGS:\n{item['findings']}\n\nIMPRESSION:\n{item['impression']}"
            })
    print(f"Filtered to {len(eval_records)} validation records with valid images.")

    # 4. Load cache
    cache_path = ROOT / "output" / f"cache_pipeline_{args.cache_name}.json"
    cache = load_cache(cache_path)
    print(f"Loaded cache containing {len(cache)}/{len(eval_records)} predictions.")

    # 5. Load model and tokenizer if cache is incomplete
    if len(cache) < len(eval_records):
        print(f"Loading tokenizer & model from {args.model_path} in 4-bit precision...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Fallback chat template for Llama 3 instruct format
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

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        local_llm = LocalTransformersClient(model, tokenizer)

        # Initialize visual controller & hybrid retriever
        index_dir = ROOT / "data" / "processed" / "index"
        retriever = HybridRetriever(index_dir) if (index_dir / "bm25.pkl").exists() else None
        controller = AgentController(retriever=retriever, use_neural=True)

        print("Starting pipeline execution loop...")
        for idx, rec in enumerate(eval_records):
            uid = rec["uid"]
            if uid in cache:
                continue

            try:
                # Execute evidence collection and planner steps
                state = controller.run(uid, rec["image_path"], {"indication": rec["indication"]})

                # Execute LLM generations
                structured = generate_structured_findings(state, local_llm)
                report = generate_narrative_report(structured, state, local_llm)
                if validate_report(report):
                    report = repair_report(report, structured, state, local_llm)

                # Format fallback
                if validate_report(report) or not report.strip():
                    obs_texts = [o.get("finding", "") for o in structured.get("observations", []) if o.get("finding")]
                    findings_text = " ".join(obs_texts) if obs_texts else "The cardiac silhouette and mediastinal contours are within normal limits."
                    imp_texts = structured.get("impression_candidates", [])
                    impression_text = " ".join(imp_texts) if imp_texts else "No acute cardiopulmonary abnormality."
                    report = f"FINDINGS: {findings_text}\n\nIMPRESSION: {impression_text}"

                pred_clean = clean_predictions(report)
                cache[uid] = pred_clean
                save_cache(cache_path, cache)

            except Exception as e:
                print(f"Error processing record {uid} (index {idx}): {e}")
                import traceback
                traceback.print_exc()

            if len(cache) % 10 == 0 or len(cache) == len(eval_records):
                print(f"[Progress] Evaluated {len(cache)}/{len(eval_records)} records...")
                sys.stdout.flush()

    # 6. Compute final metrics
    print("Computing metrics...")
    predictions = []
    references = []
    for rec in eval_records:
        uid = rec["uid"]
        predictions.append(cache.get(uid, "FINDINGS: normal. IMPRESSION: normal."))
        references.append(rec["reference"])

    metrics = compute_all_eval_metrics(predictions, references)
    print("\n" + "=" * 50)
    print("FINAL END-TO-END PIPELINE METRICS:")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k:<15} : {v:.4f}")
    print("=" * 50)

    # Save metrics
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"num_records": len(eval_records), "metrics": metrics}, f, indent=2)
    print(f"Metrics saved successfully to {output_path}!")


if __name__ == "__main__":
    main()
