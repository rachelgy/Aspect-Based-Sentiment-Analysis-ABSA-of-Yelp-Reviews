#!/usr/bin/env python
import os, argparse, yaml, pandas as pd
from src.utils import set_seed, ensure_dir
from src.preprocess import load_yelp_reviews, load_businesses, filter_food_cafes
from src.labeling import build_aspect_dataset
from src.datasets import split_and_save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--min_reviews", type=int, default=None)
    ap.add_argument("--max_reviews", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    raw_dir = cfg["paths"]["raw_dir"]
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    ensure_dir(interim_dir); ensure_dir(processed_dir)

    biz = load_businesses(raw_dir)
    cafe_biz = filter_food_cafes(biz)
    cafe_ids = set(cafe_biz["business_id"].tolist())

    max_rows = args.max_reviews or cfg["data"]["max_reviews"]
    reviews = load_yelp_reviews(raw_dir, max_rows=max_rows)
    reviews = reviews[reviews["business_id"].isin(cafe_ids)].reset_index(drop=True)

    if args.min_reviews:
        reviews = reviews.head(max(args.min_reviews, min(len(reviews), max_rows)))

    inter_path = os.path.join(interim_dir, "reviews_cafes.parquet")
    reviews.to_parquet(inter_path, index=False)
    print(f"Saved interim reviews: {inter_path} ({len(reviews)} rows)")

    keep_neutral = cfg["data"]["keep_neutral"]
    window_sentences = cfg["aspects"]["window_sentences"]
    lexicons = cfg["aspects"]["lexicons"]
    aspect_df = build_aspect_dataset(reviews, lexicons, keep_neutral, window_sentences)

    paths = split_and_save(aspect_df, processed_dir, cfg["data"]["test_size"], cfg["data"]["dev_size"], cfg["seed"])
    print("Prepared datasets for aspects:", list(paths.keys()))

if __name__ == "__main__":
    main()
