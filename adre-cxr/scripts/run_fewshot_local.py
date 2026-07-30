#!/usr/bin/env python3
"""
Few-shot report generation with a local HF model (no cloud API).

Uses the same evidence packets + Stage A schema as the ADRE pipeline, but runs
generation locally so Row 2 can be completed when Groq/Gemini quotas are exhausted.

Output: outputs/reports_fewshot/*.txt and outputs/fewshot_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.state import EvidenceState
from src.evidence.evidence_packet import EvidencePacket
from src.generation.stage_b_narrative import FEW_SHOT_EXAMPLES


INSTRUCTION = (
    "You are a board-certified radiologist writing a chest X-ray report. "
    "Use only the evidence provided. Write FINDINGS and IMPRESSION sections."
)


def build_prompt(indication: str, evidence_context: str) -> str:
    return (
        f"{INSTRUCTION}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        f"Now write the report for this study.\n"
        f"Indication: {indication or 'Not provided'}\n\n"
        f"Evidence:\n{evidence_context}\n\n"
        "FINDINGS:\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--reports-dir", default="outputs/reports_fewshot")
    parser.add_argument("--predictions", default="outputs/fewshot_predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    held = json.loads((ROOT / "data" / "processed" / "heldout_uids.json").read_text(encoding="utf-8"))
    eval_uids = [str(u) for u in held.get("eval_uids", [])][: args.limit]
    split = {
        str(r["uid"]): r
        for r in json.loads(
            (ROOT / "data" / "processed" / f"{args.split}_split.json").read_text(encoding="utf-8")
        )
    }

    reports_dir = ROOT / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = ROOT / "data" / "processed" / "evidence"

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to("cpu")
    model.eval()

    predictions = []
    for uid in tqdm(eval_uids):
        out_txt = reports_dir / f"{uid}.txt"
        ev_path = ev_dir / f"{uid}.json"
        if not ev_path.exists():
            print(f"skip {uid}: no evidence")
            continue
        packet = EvidencePacket.load(ev_path)
        rec = split.get(uid, {})
        indication = rec.get("indication", "") or packet.patient_metadata.get("indication", "")
        state = EvidenceState(patient_metadata={"indication": indication})
        state.append_evidence(packet.to_dict())
        prompt = build_prompt(indication, state.to_context())

        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        if not gen.upper().startswith("FINDINGS"):
            gen = "FINDINGS:\n" + gen
        if "IMPRESSION" not in gen.upper():
            gen = gen.rstrip() + "\n\nIMPRESSION:\nSee findings."

        out_txt.write_text(gen, encoding="utf-8")
        predictions.append({"uid": uid, "prediction": gen})

    pred_path = ROOT / args.predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} -> {pred_path}")


if __name__ == "__main__":
    main()
