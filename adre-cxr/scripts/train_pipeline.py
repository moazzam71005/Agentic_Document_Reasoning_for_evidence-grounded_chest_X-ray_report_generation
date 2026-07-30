#!/usr/bin/env python3
"""Fine-tune models on Pipeline-Generated SFT Dataset."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
import json

import torch
# Monkeypatch torch.load to bypass weights_only=True issues in PyTorch 2.6 when loading optimizer states
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

import transformers
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class PipelineSFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 2048):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_llama = "llama" in self.tokenizer.name_or_path.lower()
        
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.records[idx]
        
        # We simulate the exact two-turn conversation that happens in the pipeline
        messages = [
            # Stage A
            {"role": "user", "content": item["stage_a"]["prompt"]},
            {"role": "assistant", "content": item["stage_a"]["target"]},
            # Stage B
            {"role": "user", "content": item["stage_b"]["prompt"]},
            {"role": "assistant", "content": item["stage_b"]["target"]}
        ]
        
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        full_encoding = self.tokenizer(full_text, add_special_tokens=False)
        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]
        
        # Mask out user prompts so loss is only computed on assistant responses
        labels = list(input_ids)
        
        # Find assistant tags to dynamically mask user text
        # Varies by tokenizer (Llama vs Qwen)
        assistant_start_str = "<|start_header_id|>assistant<|end_header_id|>" if self.is_llama else "<|im_start|>assistant\n"
        assistant_end_str = "<|eot_id|>" if self.is_llama else "<|im_end|>\n"
        
        assistant_start_ids = self.tokenizer(assistant_start_str, add_special_tokens=False)["input_ids"]
        assistant_end_ids = self.tokenizer(assistant_end_str, add_special_tokens=False)["input_ids"]
        
        # Simple dynamic masking
        is_masking = True
        for i in range(len(labels)):
            # If we see assistant start, we stop masking after it
            if i + len(assistant_start_ids) <= len(labels) and labels[i:i+len(assistant_start_ids)] == assistant_start_ids:
                # Mask the start tag itself
                for j in range(i, i+len(assistant_start_ids)):
                    labels[j] = -100
                is_masking = False
                continue
                
            # If we see assistant end, we start masking again
            if i + len(assistant_end_ids) <= len(labels) and labels[i:i+len(assistant_end_ids)] == assistant_end_ids:
                is_masking = True
                # Mask the end tag itself
                for j in range(i, i+len(assistant_end_ids)):
                    labels[j] = -100
                continue
                
            if is_masking:
                labels[i] = -100
                
        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading pipeline SFT dataset from {args.sft_data}...")
    with open(args.sft_data, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records.")
    
    # Split
    rng = random.Random(42)
    rng.shuffle(data)
    train_size = int(0.9 * len(data))
    train_records = data[:train_size]
    val_records = data[train_size:]
    
    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if "llama" in args.model_id.lower() and tokenizer.chat_template is None:
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
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading base model {args.model_id}...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    train_dataset = PipelineSFTDataset(train_records, tokenizer)
    val_dataset = PipelineSFTDataset(val_records, tokenizer)

    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    output_dir = ROOT / args.output_dir
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
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n=== STARTING SUPERVISED FINE-TUNING ===")
    trainer.train()

    adapter_dir = output_dir.with_name(output_dir.name + "-lora")
    print(f"Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print("\n=== MERGING BASE MODEL AND LORA ADAPTER ===")
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading base model in half precision for merging...")
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print("Loading adapter model...")
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    merged_output_dir = output_dir.with_name(output_dir.name + "-merged")
    print(f"Saving merged model to {merged_output_dir}...")
    merged_model.save_pretrained(str(merged_output_dir))
    tokenizer.save_pretrained(str(merged_output_dir))
    print("Completed successfully!")

if __name__ == "__main__":
    main()
