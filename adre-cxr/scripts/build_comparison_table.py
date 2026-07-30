#!/usr/bin/env python3
"""Build the 3-row Zero-shot / Few-shot / LoRA comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import compute_all_metrics


ROWS = [
    ("Zero-shot", "outputs/dev_predictions.jsonl"),
    ("Few-shot", "outputs/fewshot_predictions.jsonl"),
    ("LoRA", "outputs/lora_predictions.jsonl"),
]


def load_pairs(pred_path: Path, split: list[dict]) -> tuple[list[str], list[str]]:
    gt = {str(r["uid"]): (r["findings"] + "\n\n" + r["impression"]) for r in split}
    preds, refs = [], []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["uid"])
        if uid in gt:
            preds.append(row["prediction"])
            refs.append(gt[uid])
    return preds, refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output",
        default="outputs/comparison_table.json",
        help="JSON results path",
    )
    parser.add_argument(
        "--md-output",
        default="outputs/comparison_table.md",
        help="Markdown table path",
    )
    args = parser.parse_args()

    import yaml

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split = json.loads(
        (ROOT / "data" / "processed" / f"{args.split}_split.json").read_text(
            encoding="utf-8"
        )
    )
    bert_model = cfg["evaluation"]["bertscore_model"]

    table: dict[str, dict] = {}
    missing = []
    for name, rel in ROWS:
        path = ROOT / rel
        if not path.exists():
            missing.append(f"{name}: missing {rel}")
            continue
        preds, refs = load_pairs(path, split)
        if not preds:
            missing.append(f"{name}: empty predictions in {rel}")
            continue
        metrics = compute_all_metrics(preds, refs, bert_model)
        metrics["n"] = len(preds)
        table[name] = metrics
        print(f"{name} (N={len(preds)}): {json.dumps(metrics, indent=2)}")

    out_json = ROOT / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(table, indent=2), encoding="utf-8")

    cols = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor", "bertscore_f1"]
    headers = ["Setting", "N", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "BERTScore F1"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, _ in ROWS:
        if name not in table:
            lines.append(f"| {name} | — | " + " | ".join(["—"] * 7) + " |")
            continue
        m = table[name]
        cells = [name, str(m["n"])] + [f"{m[c]:.4f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")

    md = "\n".join(lines) + "\n"
    md_path = ROOT / args.md_output
    md_path.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Wrote {out_json}")
    print(f"Wrote {md_path}")
    if missing:
        print("\nMissing rows:")
        for msg in missing:
            print(f"  - {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
