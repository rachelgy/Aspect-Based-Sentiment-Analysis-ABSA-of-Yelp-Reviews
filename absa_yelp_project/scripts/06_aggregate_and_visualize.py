#!/usr/bin/env python
import os, json, yaml
import pandas as pd
from joblib import load
from src.aggregate import predict_sklearn, aggregate_business_level
from src.visualize import bar_top_issues, heatmap_aspects

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    aspects = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    all_preds = []
    for aspect in aspects:
        test_path = os.path.join(processed_dir, aspect, "test.parquet")
        df = pd.read_parquet(test_path)
        model_path = os.path.join(models_dir, f"lr_{aspect}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f"xgb_{aspect}.joblib")
        if not os.path.exists(model_path):
            print(f"Skipping {aspect} — no trained sklearn model found.")
            continue
        preds = predict_sklearn(model_path, df["text"])
        sub = pd.DataFrame({"business_id": df["business_id"], "aspect": aspect, "pred": preds})
        all_preds.append(sub)
    if not all_preds:
        print("No predictions to aggregate.")
        return
    all_preds = pd.concat(all_preds, ignore_index=True)
    agg = aggregate_business_level(all_preds)
    out_csv = os.path.join(cfg["paths"]["reports_dir"], "business_aspect_scores.csv")
    agg.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    bar_top_issues(agg, figures_dir)
    heatmap_aspects(agg, figures_dir)
    print(f"Charts saved under {figures_dir}")

if __name__ == "__main__":
    main()
