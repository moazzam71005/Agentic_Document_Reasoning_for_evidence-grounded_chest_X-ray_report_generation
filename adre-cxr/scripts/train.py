#!/usr/bin/env python3
"""Fine-tune Qwen2.5-3B using LoRA / QLoRA optimized for 6GB VRAM GPUs."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import json

import pandas as pd
import torch
# Monkeypatch torch.load to bypass weights_only=True issues in PyTorch 2.6 when loading optimizer states
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

import yaml
from torch.utils.data import Dataset

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep.parse_reports import parse_reports_df
from src.eval.metrics import compute_all_metrics

# Monkey patch for transformers tokenizer in bert_score
import transformers
if not hasattr(transformers.PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1
    transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens = build_inputs_with_special_tokens


class CXRReportDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = (
            "You are a board-certified radiologist writing a chest X-ray report. "
            "Write two sections exactly: FINDINGS: and IMPRESSION:. "
            "Rules: Follow standard radiology phrasing."
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.records[idx]
        input_text = f"Patient Indication: {item['indication']}\nComparison: {item['comparison']}"
        output_text = f"FINDINGS:\n{item['findings']}\n\nIMPRESSION:\n{item['impression']}"

        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]

        # Use tokenizer apply_chat_template
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        prompt_encoding = self.tokenizer(prompt_text, add_special_tokens=False)
        full_encoding = self.tokenizer(full_text, add_special_tokens=False)

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]
        
        prompt_len = len(prompt_encoding["input_ids"])

        # Create labels with masking for prompt tokens (mask with -100)
        # BPE alignment check
        assistant_header = "<|im_start|>assistant\n"
        header_ids = self.tokenizer(assistant_header, add_special_tokens=False)["input_ids"]
        
        match_idx = -1
        for i in range(len(input_ids) - len(header_ids) + 1):
            if input_ids[i : i + len(header_ids)] == header_ids:
                match_idx = i + len(header_ids)
                break

        if match_idx != -1:
            labels = [-100] * match_idx + input_ids[match_idx:]
        else:
            labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Truncate to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        # Ensure correct torch dtypes
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def evaluate_model(
    model,
    tokenizer,
    records: list[dict],
    limit: int | None = None,
    bertscore_model: str = "distilbert-base-uncased",
) -> dict[str, float]:
    """Evaluate model predictions on evaluation records against references."""
    model.eval()
    if limit is not None:
        records = records[:limit]

    instruction = (
        "You are a board-certified radiologist writing a chest X-ray report. "
        "Write two sections exactly: FINDINGS: and IMPRESSION:. "
        "Rules: Follow standard radiology phrasing."
    )

    predictions = []
    references = []

    print(f"Generating predictions for {len(records)} records...")
    for idx, item in enumerate(records):
        input_text = f"Patient Indication: {item['indication']}\nComparison: {item['comparison']}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Skip prompt tokens
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append(prediction)
        
        ref_text = f"FINDINGS:\n{item['findings']}\n\nIMPRESSION:\n{item['impression']}"
        references.append(ref_text)

        if idx < 3:
            print(f"\n--- Sample {idx+1} ---")
            print(f"Reference:\n{ref_text}")
            print(f"Prediction:\n{prediction}")
            print("-" * 20)

    # Compute evaluation metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(predictions, references, bertscore_model)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning script for Qwen2.5-3B")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick SFT dry-run test without training")
    parser.add_argument("--eval-only", action="store_true", help="Only run baseline evaluation on validation split")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit number of validation records for evaluation speed")
    parser.add_argument("--epochs", type=int, default=3, help="Number of SFT training epochs")
    parser.add_argument("--save-steps", type=int, default=100, help="Validate and save every N optimizer steps")
    parser.add_argument("--skip-merge", action="store_true", help="Keep the best LoRA adapter without writing a second merged base model")
    parser.add_argument(
        "--split-mode",
        choices=["heldout-500", "full-90-10"],
        default="heldout-500",
        help=(
            "heldout-500 excludes the frozen 100/400 ADRE studies; "
            "full-90-10 trains on a deterministic 90%% split of every valid report"
        ),
    )
    args = parser.parse_args()

    # Load project config
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    data_root = ROOT / cfg["paths"]["data_root"]
    reports_csv = data_root / "indiana_reports.csv"
    if not reports_csv.exists():
        print(f"Error: indiana_reports.csv not found at {reports_csv}")
        sys.exit(1)

    print(f"Loading and parsing reports from {reports_csv}...")
    reports = pd.read_csv(reports_csv)
    parsed = parse_reports_df(
        reports,
        min_findings=cfg["data"]["min_findings_chars"],
        min_impression=cfg["data"]["min_impression_chars"],
    )
    print(f"Total valid reports: {len(parsed)}")

    # Reproducible splits mapping prepare_data.py
    rng = random.Random(42)
    shuffled = parsed.copy()
    rng.shuffle(shuffled)

    # The default protects the frozen ADRE dev/test studies from training leakage.
    # full-90-10 is available for deployment-oriented training where that frozen
    # split will not subsequently be reported as an unseen test set.
    sft_records = shuffled if args.split_mode == "full-90-10" else shuffled[500:]
    train_size = int(0.9 * len(sft_records))
    train_records = sft_records[:train_size]
    val_records = sft_records[train_size:]
    print(
        f"Split mode: {args.split_mode} | "
        f"SFT Train: {len(train_records)}, Validation: {len(val_records)}"
    )

    # Load model and tokenizer
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Config (QLoRA)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading quantized base model {model_id}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Evaluate dry-run or eval-only baseline
    bertscore_model = cfg["evaluation"].get("bertscore_model", "distilbert-base-uncased")
    if args.eval_only:
        print("\n=== RUNNING BASELINE EVALUATION ONLY ===")
        metrics = evaluate_model(model, tokenizer, val_records, limit=args.eval_limit, bertscore_model=bertscore_model)
        print("Baseline Metrics:", json.dumps(metrics, indent=2))
        sys.exit(0)

    if args.dry_run:
        print("\n=== SFT DRY-RUN VERIFICATION ===")
        dataset = CXRReportDataset(train_records[:5], tokenizer)
        item = dataset[0]
        print("Input IDs shape:", item["input_ids"].shape)
        print("Labels shape:", item["labels"].shape)
        
        # Verify prompt tokens are indeed masked
        non_masked_count = (item["labels"] != -100).sum().item()
        print(f"Masked tokens (-100): {len(item['labels']) - non_masked_count}")
        print(f"Loss tokens (active): {non_masked_count}")
        print("Dry run completed successfully.")
        sys.exit(0)

    # Wrap with PEFT LoRA Config
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("Wrapping model with LoRA adapter...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for 6GB VRAM optimization
    model.gradient_checkpointing_enable()

    # Create dataset objects
    train_dataset = CXRReportDataset(train_records, tokenizer)
    val_dataset = CXRReportDataset(val_records, tokenizer)

    # Define SFT Data Collator (padding and masking)
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    output_dir = ROOT / "weights" / "qwen2.5-3b-ft-checkpoints"
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False
    )

    # Pre-train baseline evaluation on subset for comparison
    print("\n=== COMPUTING BASELINE METRICS BEFORE TRAINING ===")
    baseline_metrics = evaluate_model(model, tokenizer, val_records, limit=args.eval_limit, bertscore_model=bertscore_model)
    print("Baseline Metrics:", json.dumps(baseline_metrics, indent=2))

    # Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("\n=== STARTING SUPERVISED FINE-TUNING ===")
    checkpoint_dir = str(output_dir)
    resume = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            resume = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"Found existing checkpoints. Resuming training from: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    print(f"Best validation checkpoint: {trainer.state.best_model_checkpoint}")

    # Save adapter
    adapter_dir = ROOT / "weights" / "qwen2.5-3b-lora"
    print(f"Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Evaluate model post-training
    print("\n=== COMPUTING POST-FINE-TUNING METRICS ===")
    ft_metrics = evaluate_model(model, tokenizer, val_records, limit=args.eval_limit, bertscore_model=bertscore_model)
    print("Post-Fine-Tuning Metrics:", json.dumps(ft_metrics, indent=2))

    # Summary table
    print("\n" + "=" * 50)
    print("FINE-TUNING COMPARATIVE SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<15} | {'Baseline':<12} | {'Fine-Tuned':<12}")
    print("-" * 50)
    for m in ["bleu4", "rouge_l", "meteor", "bertscore_f1"]:
        print(f"{m:<15} | {baseline_metrics.get(m, 0.0):<12.4f} | {ft_metrics.get(m, 0.0):<12.4f}")
    print("=" * 50)

    if args.skip_merge:
        print(
            "\nSkipping merged-model export. The best LoRA adapter and tokenizer "
            f"are saved at {adapter_dir}."
        )
        return

    # Merging Base Model + Adapter (requires loading unquantized model for weight merging)
    print("\n=== MERGING BASE MODEL AND LORA ADAPTER ===")
    # Free memory
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load base model in 16-bit to perform the weight merge
    print("Loading base model in half precision for merging...")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to stay safely within 6GB VRAM
        trust_remote_code=True
    )
    
    from peft import PeftModel
    print("Loading adapter model...")
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    
    print("Merging weights...")
    merged_model = merged_model.merge_and_unload()

    merged_output_dir = ROOT / "weights" / "qwen2.5-3b-merged"
    print(f"Saving merged model to {merged_output_dir}...")
    merged_model.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))
    print("Model fine-tuning and merging successfully completed!")


if __name__ == "__main__":
    main()
