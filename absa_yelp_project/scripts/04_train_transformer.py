#!/usr/bin/env python
import os, json, yaml, argparse
from src.models_transformer import train_transformer_for_aspect

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    hp = cfg["transformer"]
    if args.epochs is not None:
        hp["epochs"] = int(args.epochs)

    aspects = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    results = []
    for aspect in aspects:
        paths = {
            "train": os.path.join(processed_dir, aspect, "train.parquet"),
            "dev": os.path.join(processed_dir, aspect, "dev.parquet"),
            "test": os.path.join(processed_dir, aspect, "test.parquet"),
        }
        res = train_transformer_for_aspect(paths, models_dir, aspect, hp["model_name"], hp["max_length"], hp["lr"], hp["batch_size"], hp["epochs"], hp["warmup_ratio"], seed=cfg["seed"], no_cuda=args.no_cuda)
        results.append(res)
        print(f"Trained HF for {aspect}: {res['metrics_dev']['f1']:.3f} (dev F1)")
    os.makedirs(cfg["paths"]["metrics_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["metrics_dir"], "transformer_dev_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
