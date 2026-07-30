#!/usr/bin/env python3
"""
QLoRA / LoRA fine-tune for ADRE-CXR Row 3.

Preferred: run notebooks/02_lora_finetune.ipynb on Kaggle GPU.
This script mirrors that notebook for local or cloud use.

Example (GPU):
  python scripts/run_lora_train.py --max-train 2000 --epochs 3

Example (CPU smoke / slow):
  python scripts/run_lora_train.py --max-train 200 --epochs 1 --batch-size 1 --no-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-csv", default="data/raw/indiana_reports.csv")
    parser.add_argument("--heldout", default="data/processed/heldout_uids.json")
    parser.add_argument("--output-predictions", default="outputs/lora_predictions.jsonl")
    parser.add_argument("--adapter-dir", default="outputs/lora_adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from tqdm.auto import tqdm

    held = json.loads((ROOT / args.heldout).read_text(encoding="utf-8"))
    held_uids = set(map(str, held["dev"])) | set(map(str, held["test"]))
    eval_uids = [str(u) for u in held.get("eval_uids", held["dev"][:26])]

    df = pd.read_csv(ROOT / args.reports_csv)
    df["uid"] = df["uid"].astype(str)
    for col in ("findings", "impression", "indication"):
        df[col] = df[col].fillna("").astype(str)
    usable = df[(df["findings"].str.len() >= 10) & (df["impression"].str.len() >= 5)].copy()
    train_df = usable[~usable["uid"].isin(held_uids)].head(args.max_train)
    eval_df = usable[usable["uid"].isin(eval_uids)].copy()
    eval_df["_ord"] = eval_df["uid"].map({u: i for i, u in enumerate(eval_uids)})
    eval_df = eval_df.sort_values("_ord").drop(columns="_ord")

    instruction = (
        "You are a radiologist. Given the clinical indication, write a chest X-ray report "
        "with exactly two sections: FINDINGS and IMPRESSION."
    )

    def target(f: str, i: str) -> str:
        return f"FINDINGS:\n{f.strip()}\n\nIMPRESSION:\n{i.strip()}"

    train_rows = []
    for _, r in train_df.iterrows():
        train_rows.append(
            {
                "messages": [
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": f"Indication: {r['indication'].strip() or 'Not provided'}",
                    },
                    {
                        "role": "assistant",
                        "content": target(r["findings"], r["impression"]),
                    },
                ]
            }
        )
    train_ds = Dataset.from_list(train_rows)
    print(f"Train={len(train_ds)} Eval={len(eval_df)} Model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.no_4bit or not torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.to("cpu")
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    adapter_dir = ROOT / args.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    use_cpu = not torch.cuda.is_available()
    sft_args = SFTConfig(
        output_dir=str(ROOT / "outputs" / "sft_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        use_cpu=use_cpu,
        max_length=args.max_length,
        packing=False,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    model.eval()
    predictions = []
    device = next(model.parameters()).device
    for _, r in tqdm(eval_df.iterrows(), total=len(eval_df)):
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": f"Indication: {(r['indication'] or 'Not provided').strip()}",
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        if "FINDINGS:" not in gen.upper():
            gen = "FINDINGS:\n" + gen
        predictions.append({"uid": str(r["uid"]), "prediction": gen})

    out_path = ROOT / args.output_predictions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
