import os
import sys
import json
import random
import argparse
from pathlib import Path
import pandas as pd
import torch
import transformers
from tqdm import tqdm

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Bypass transformers PyTorch version restriction check for loading legacy .bin weights
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from src.data_prep.parse_reports import parse_reports_df
from src.agent.controller import AgentController
from src.knowledge_index.retrieve import HybridRetriever
from src.generation.stage_a_findings import generate_structured_findings, STAGE_A_SCHEMA

class LocalTransformersClient:
    """Wrapper around local transformers model to match LLMClient interface."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.provider = "local_transformers"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0, top_p: float = 1.0, json_mode: bool = False) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256 if json_mode else 512,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def load_progress(output_path: Path) -> list[dict]:
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_progress(output_path: Path, data: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local weights path to Llama-3-OpenBioLLM-8B")
    parser.add_argument("--output-json", default="output/pipeline_sft_data.json", help="Output path for SFT dataset")
    args = parser.parse_args()

    # Load splits config
    reports_csv = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "indiana_reports.csv"
    projections_csv = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "indiana_projections.csv"
    
    if not reports_csv.exists() or not projections_csv.exists():
        print("Error: reports or projections csv not found!")
        sys.exit(1)

    print("Loading and parsing reports...")
    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)
    
    parsed = parse_reports_df(reports, min_findings=10, min_impression=10)
    print(f"Total valid reports: {len(parsed)}")

    # Reproducible splits mapping train_llama.py
    rng = random.Random(42)
    shuffled = list(parsed)
    rng.shuffle(shuffled)
    sft_records = shuffled[500:]
    print(f"SFT records: {len(sft_records)}")

    # Load progress
    output_path = Path(args.output_json)
    sft_data = load_progress(output_path)
    processed_uids = {item["uid"] for item in sft_data}
    print(f"Loaded progress: {len(processed_uids)}/{len(sft_records)} processed.")

    if len(processed_uids) >= len(sft_records):
        print("All records already processed!")
        return

    # Initialize Retriever & Controller
    print("Initializing Retriever & Controller...")
    retriever = HybridRetriever(ROOT / "data" / "processed" / "index")
    controller = AgentController(retriever=retriever, use_neural=True)

    # Initialize tokenizer & model if progress is incomplete
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

    print("Starting data generation loop...")
    for idx, rec in enumerate(sft_records):
        uid = rec["uid"]
        if uid in processed_uids:
            continue

        # Resolve image path
        proj = projections[projections["uid"] == int(uid)]
        if proj.empty:
            continue
        frontal = proj[proj["projection"] == "Frontal"] if "Frontal" in proj["projection"].values else proj
        row = frontal.iloc[0]
        img_path = ROOT / "data" / "raw" / "chest-xrays-indiana-university" / "images" / "images_normalized" / row["filename"]
        if not img_path.exists():
            continue

        try:
            # 1. Run pipeline controller to gather evidence state
            state = controller.run(uid, str(img_path), {"indication": rec.get("indication", "")})

            # 2. Stage A structured findings generation
            structured = generate_structured_findings(state, local_llm)

            # Define prompts matching pipeline exactly
            stage_a_prompt = f"IMPORTANT: Do NOT output any thinking process, internal monologue, or '<think>' tags. Do NOT write any reasoning steps. Skip thinking entirely and output ONLY the raw JSON object matching the schema below.\n\nYou are a radiology evidence extraction agent.\nGiven the evidence state below, produce ONLY a JSON object matching this schema:\n{STAGE_A_SCHEMA}\n\nRules:\n- Every observation must cite an evidence_ref from the state\n- Do NOT invent findings not supported by evidence\nEvidence state:\n{state.to_context()}\n"
            stage_b_prompt = f"IMPORTANT: Do NOT output any thinking process, internal monologue, or '<think>' tags. Do NOT write any reasoning steps. Skip thinking entirely and output ONLY the text report.\n\nYou are a board-certified radiologist writing a chest X-ray report.\n\nWrite two sections exactly:\nFINDINGS:\n<detailed findings paragraph>\n\nIMPRESSION:\n<concise impression paragraph>\n\nRules:\n- Use only findings supported by the structured evidence below\n- Follow standard radiology phrasing from retrieved guidelines\n- Do not add unsupported diagnoses\n\nStructured findings:\n{structured}\n\nEvidence and guidelines:\n{state.to_context()}\n"

            ground_truth_report = f"FINDINGS:\n{rec['findings']}\n\nIMPRESSION:\n{rec['impression']}"

            sft_data.append({
                "uid": uid,
                "stage_a": {
                    "prompt": stage_a_prompt,
                    "target": json.dumps(structured, ensure_ascii=False)
                },
                "stage_b": {
                    "prompt": stage_b_prompt,
                    "target": ground_truth_report
                }
            })

            # Save progress every 10 records
            if len(sft_data) % 10 == 0:
                print(f"[Progress] Generated {len(sft_data)}/{len(sft_records)} SFT pairs...")
                save_progress(output_path, sft_data)
                sys.stdout.flush()

        except Exception as e:
            print(f"Error generating SFT data for {uid} (index {idx}): {e}")

    # Final save
    save_progress(output_path, sft_data)
    print("Successfully generated all pipeline SFT data!")

if __name__ == "__main__":
    main()
