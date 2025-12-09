from __future__ import annotations
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save(df: pd.DataFrame, processed_dir: str, test_size: float, dev_size: float, seed: int):
    os.makedirs(processed_dir, exist_ok=True)
    paths = {}
    for aspect in sorted(df["aspect"].unique()):
        sub = df[df["aspect"] == aspect].copy()
        if sub["label"].nunique() < 2 or len(sub) < 50:
            continue
        train_df, temp_df = train_test_split(sub, test_size=test_size + dev_size, random_state=seed, stratify=sub["label"])
        rel_dev = dev_size / (test_size + dev_size)
        dev_df, test_df = train_test_split(temp_df, test_size=1 - rel_dev, random_state=seed, stratify=temp_df["label"])
        aspect_dir = os.path.join(processed_dir, aspect)
        os.makedirs(aspect_dir, exist_ok=True)
        train_p = os.path.join(aspect_dir, "train.parquet")
        dev_p = os.path.join(aspect_dir, "dev.parquet")
        test_p = os.path.join(aspect_dir, "test.parquet")
        train_df.to_parquet(train_p, index=False)
        dev_df.to_parquet(dev_p, index=False)
        test_df.to_parquet(test_p, index=False)
        paths[aspect] = {"train": train_p, "dev": dev_p, "test": test_p}
    return paths
