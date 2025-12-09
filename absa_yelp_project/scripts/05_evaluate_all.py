#!/usr/bin/env python
import os, json, yaml
from glob import glob
from src.eval_models import eval_sklearn_model

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    metrics_dir = cfg["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    aspects = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])

    # Eval LR + XGB (sklearn pipelines)
    for model_prefix in ["lr_", "xgb_"]:
        all_metrics = {}
        for aspect in aspects:
            paths = {
                "train": os.path.join(processed_dir, aspect, "train.parquet"),
                "dev": os.path.join(processed_dir, aspect, "dev.parquet"),
                "test": os.path.join(processed_dir, aspect, "test.parquet"),
            }
            model_path = os.path.join(models_dir, f"{model_prefix}{aspect}.joblib")
            if not os.path.exists(model_path):
                continue
            rep = eval_sklearn_model(paths, model_path)
            all_metrics[aspect] = rep
            print(f"{model_prefix}{aspect} — test F1 (weighted): {rep['weighted avg']['f1']:.3f}")
        out = os.path.join(metrics_dir, f"{model_prefix.strip('_')}_test_metrics.json")
        with open(out, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
