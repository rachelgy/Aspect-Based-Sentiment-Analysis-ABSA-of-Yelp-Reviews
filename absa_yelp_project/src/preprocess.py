from __future__ import annotations
import os, re
import pandas as pd
from typing import List, Dict, Tuple
from .io import stream_json

def clean_text(s: str) -> str:
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_yelp_reviews(raw_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    review_path = os.path.join(raw_dir, "yelp_academic_dataset_review.json")
    rows = []
    for i, obj in enumerate(stream_json(review_path)):
        rows.append({
            "review_id": obj.get("review_id"),
            "user_id": obj.get("user_id"),
            "business_id": obj.get("business_id"),
            "stars": obj.get("stars"),
            "text": clean_text(obj.get("text", ""))
        })
        if max_rows is not None and i + 1 >= max_rows:
            break
    return pd.DataFrame(rows)

def load_businesses(raw_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    biz_path = os.path.join(raw_dir, "yelp_academic_dataset_business.json")
    rows = []
    for i, obj in enumerate(stream_json(biz_path)):
        rows.append({
            "business_id": obj.get("business_id"),
            "name": obj.get("name"),
            "city": obj.get("city"),
            "state": obj.get("state"),
            "stars": obj.get("stars"),
            "review_count": obj.get("review_count"),
            "categories": obj.get("categories"),
            "attributes": obj.get("attributes", {}),
            "price": obj.get("attributes", {}).get("RestaurantsPriceRange2", None),
        })
        if max_rows is not None and i + 1 >= max_rows:
            break
    return pd.DataFrame(rows)

def filter_food_cafes(biz: pd.DataFrame) -> pd.DataFrame:
    target = ("Coffee & Tea", "Cafes", "Bakeries")
    def is_target(cats: str) -> bool:
        if not isinstance(cats, str):
            return False
        return any(t in cats for t in target)
    return biz[biz["categories"].apply(is_target)].copy()
