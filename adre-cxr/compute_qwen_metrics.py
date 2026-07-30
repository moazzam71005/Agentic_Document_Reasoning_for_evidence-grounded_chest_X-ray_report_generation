import json
import argparse
import pandas as pd
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.eval.metrics import compute_all_metrics
from src.data_prep.parse_reports import parse_reports_df

ROOT = Path(".")

def evaluate_cached(cache_file, output_json, bertscore_model='roberta-large'):
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    data_dir = ROOT / "data" / "raw" / "chest-xrays-indiana-university"
    reports_csv = data_dir / "indiana_reports.csv"
    projections_csv = data_dir / "indiana_projections.csv"
    images_dir = data_dir / "images" / "images_normalized"

    df_raw = pd.read_csv(str(reports_csv))
    reports = parse_reports_df(df_raw)
    train_records, val_records = train_test_split(reports, test_size=0.1, random_state=42)
    
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

    eval_records = []
    for item in val_records:
        uid_str = str(item["uid"])
        if uid_str in image_map:
            eval_records.append({
                "uid": uid_str,
                "reference": f"FINDINGS:\n{item['findings']}\n\nIMPRESSION:\n{item['impression']}"
            })

    predictions = []
    references = []
    
    for rec in eval_records:
        uid = str(rec['uid'])
        if uid in cache:
            predictions.append(cache[uid])
            references.append(rec['reference'])
            
    print(f'Computing metrics for {len(predictions)} records...')
    metrics = compute_all_metrics(predictions, references, bertscore_model=bertscore_model)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print('Done!', metrics)

if __name__ == '__main__':
    evaluate_cached('output/cache_pipeline_qwen_pipeline.json', 'output/full_pipeline_metrics_qwen.json')
