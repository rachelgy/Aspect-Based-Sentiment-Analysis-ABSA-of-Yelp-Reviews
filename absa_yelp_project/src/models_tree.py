from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from joblib import dump

def train_xgb_for_aspect(paths: Dict[str, str], models_dir: str, aspect: str) -> Dict:
    os.makedirs(models_dir, exist_ok=True)
    train = pd.read_parquet(paths["train"])
    dev = pd.read_parquet(paths["dev"])
    X_train, y_train = train["text"].tolist(), train["label"].tolist()
    X_dev, y_dev = dev["text"].tolist(), dev["label"].tolist()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=60000)),
        ("clf", XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="logloss"
        ))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_dev)
    report = classification_report(y_dev, preds, output_dict=True)
    model_path = os.path.join(models_dir, f"xgb_{aspect}.joblib")
    dump(pipe, model_path)
    return {"aspect": aspect, "metrics_dev": report, "model_path": model_path}
