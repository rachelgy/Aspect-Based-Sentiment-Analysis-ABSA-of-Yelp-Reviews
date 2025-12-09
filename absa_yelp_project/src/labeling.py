from __future__ import annotations
import re
import nltk
from typing import Dict, List, Tuple
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    return SENT_SPLIT.split(text)

def match_aspects(text: str, lexicons: Dict[str, List[str]], window_sentences: int = 1) -> List[Tuple[str, str]]:
    sents = split_sentences(text)
    spans = []
    for i, s in enumerate(sents):
        low = s.lower()
        for aspect, kws in lexicons.items():
            if any(kw in low for kw in kws):
                start = max(0, i - window_sentences)
                end = min(len(sents), i + window_sentences + 1)
                span = " ".join(sents[start:end]).strip()
                spans.append((aspect, span))
    return spans

def weak_label_sentiment(spans: List[str], sia: SentimentIntensityAnalyzer, keep_neutral: bool = False) -> List[int]:
    labels = []
    for t in spans:
        score = sia.polarity_scores(t)["compound"]
        if score > 0.05:
            labels.append(1)
        elif score < -0.05:
            labels.append(0)
        else:
            if keep_neutral:
                labels.append(2)
            else:
                labels.append(-1)
    return labels

def build_aspect_dataset(df: pd.DataFrame, lexicons: Dict[str, List[str]], keep_neutral: bool, window_sentences: int) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    rows = []
    for _, r in df.iterrows():
        spans = match_aspects(r["text"], lexicons, window_sentences)
        if not spans:
            continue
        aspects, texts = zip(*spans)
        labels = weak_label_sentiment(list(texts), sia, keep_neutral=keep_neutral)
        for aspect, span_text, lab in zip(aspects, texts, labels):
            if lab == -1:
                continue
            rows.append({
                "review_id": r["review_id"],
                "business_id": r["business_id"],
                "aspect": aspect,
                "text": span_text,
                "label": lab,
            })
    return pd.DataFrame(rows)
