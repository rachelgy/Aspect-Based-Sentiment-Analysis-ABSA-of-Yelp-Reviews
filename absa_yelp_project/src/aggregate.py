from __future__ import annotations
import os, json
import pandas as pd
from typing import Dict
from joblib import load

def predict_sklearn(pipe_path: str, texts: pd.Series) -> pd.Series:
    pipe = load(pipe_path)
    return pd.Series(pipe.predict(texts.tolist()))

def aggregate_business_level(aspect_preds: pd.DataFrame) -> pd.DataFrame:
    agg = (aspect_preds
           .groupby(["business_id", "aspect"])["pred"]
           .mean()
           .reset_index()
           .pivot(index="business_id", columns="aspect", values="pred")
           .fillna(0.0)
           .reset_index())
    return agg
