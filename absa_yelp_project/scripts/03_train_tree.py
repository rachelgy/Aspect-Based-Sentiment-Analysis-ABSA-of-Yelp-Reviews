#!/usr/bin/env python
import os, json, yaml
from src.models_tree import train_xgb_for_aspect

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    aspects = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    results = []
    for aspect in aspects:
        paths = {
            "train": os.path.join(processed_dir, aspect, "train.parquet"),
            "dev": os.path.join(processed_dir, aspect, "dev.parquet"),
            "test": os.path.join(processed_dir, aspect, "test.parquet"),
        }
        res = train_xgb_for_aspect(paths, models_dir, aspect)
        results.append(res)
        print(f"Trained XGB for {aspect}: {res['metrics_dev']['weighted avg']['f1']:.3f} (dev F1)")
    os.makedirs(cfg["paths"]["metrics_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["metrics_dir"], "xgb_dev_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
