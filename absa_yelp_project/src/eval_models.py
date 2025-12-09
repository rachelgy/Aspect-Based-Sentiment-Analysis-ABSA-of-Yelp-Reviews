from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict
from sklearn.metrics import classification_report
from joblib import load

def eval_sklearn_model(paths: Dict[str, str], model_path: str) -> Dict:
    pipe = load(model_path)
    test = pd.read_parquet(paths["test"])
    preds = pipe.predict(test["text"].tolist())
    report = classification_report(test["label"].tolist(), preds, output_dict=True)
    return report
