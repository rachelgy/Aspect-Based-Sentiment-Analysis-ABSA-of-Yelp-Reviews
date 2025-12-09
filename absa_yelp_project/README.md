# Yelp ABSA (Aspect-Based Sentiment Analysis) — Minimal Project

This repository contains a *practical* scaffold to run an end-to-end ABSA pipeline on the Yelp Open Dataset.

**Tracks the proposal exactly:**  
- Aspects: `service`, `food`, `ambience`, `pricing`  
- Models: Logistic Regression (TF–IDF), XGBoost/CatBoost, DistilBERT  
- Metrics: accuracy, precision, recall, F1  
- Outputs: per-aspect predictions, business-level aggregation, simple visuals

> ⚠️ **Data note**: Yelp Open Dataset is large. Download the JSON files manually from Yelp and place them under `data/raw/`:
> - `yelp_academic_dataset_review.json` (required)
> - `yelp_academic_dataset_business.json` (recommended for business categories/price)
> You can subset by category (e.g., "Coffee & Tea", "Cafes", "Bakeries") during preprocessing to keep compute small.

## Quickstart

```bash
# 1) Python env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Put Yelp JSON here:
#   data/raw/yelp_academic_dataset_review.json
#   data/raw/yelp_academic_dataset_business.json

# 3) Preprocess + weak-label aspects
python scripts/01_preprocess_and_label.py --min_reviews 20000 --max_reviews 120000 --seed 42

# 4) Train models
python scripts/02_train_baseline.py
python scripts/03_train_tree.py
python scripts/04_train_transformer.py --epochs 2  # increase if you have GPU/time

# 5) Evaluate + compare
python scripts/05_evaluate_all.py

# 6) Aggregate to business-level & visualize
python scripts/06_aggregate_and_visualize.py
```

## Project layout

```
absa_yelp_project/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                       # put Yelp JSON files here
│   ├── interim/                   # parquet/csv intermediates
│   └── processed/                 # train/dev/test splits per aspect
├── models/                        # saved weights/artifacts
├── reports/
│   ├── figures/                   # charts
│   └── metrics/                   # JSON/CSV metrics
├── scripts/                       # entrypoints (runnable)
└── src/                           # library code (importable)
```

## What this pipeline does

1. **Preprocess**: loads Yelp JSON (streaming), filters to restaurants/cafés, cleans text.
2. **Aspect detection (weak supervision)**: uses lightweight keyword lexicons to detect whether a review mentions an aspect; extracts sentence spans around those mentions.
3. **Sentiment labeling (weak)**: uses VADER polarity on the local sentence/span to derive a binary sentiment label (pos/neg). Neutral is dropped by default; you can keep it.
4. **Datasets**: builds per-aspect datasets `(text, label)` with train/dev/test splits.
5. **Models**:
   - **LR (TF–IDF)** baseline
   - **XGBoost/CatBoost** with TF–IDF + lexicon features
   - **DistilBERT** fine-tuned via Hugging Face `transformers`
6. **Eval**: reports accuracy/precision/recall/F1 per aspect and macro-average.
7. **Aggregation**: rolls predictions up to business-level to show which aspects drive ratings.
8. **Visuals**: a few clean matplotlib charts for the deck.

## Notes & tips

- To run small, limit reviews via `--max_reviews` flag.
- You can expand lexicons in `configs/config.yaml` → `aspects.lexicons` for higher recall.
- If GPU available, `transformers` will auto-detect. Otherwise, set `--no_cuda` in `04_train_transformer.py`.

---

*Built for a 2–3 week academic project: simple, documented, and easy to iterate.*
