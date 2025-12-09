import os
import pandas as pd
import matplotlib.pyplot as plt

def bar_top_issues(agg_df: pd.DataFrame, out_dir: str, top_n: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    agg_df["issue_score"] = 1 - agg_df.drop(columns=["business_id"]).mean(axis=1)
    top = agg_df.sort_values("issue_score", ascending=False).head(top_n)
    plt.figure()
    plt.bar(range(len(top)), top["issue_score"].values)
    plt.xticks(range(len(top)), top["business_id"].values, rotation=90)
    plt.title("Top Businesses by Overall Negative Aspect Signal")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_business_issues.png"), dpi=160)

def heatmap_aspects(agg_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    aspects = [c for c in agg_df.columns if c not in ["business_id", "issue_score"]]
    data = agg_df[aspects].values
    plt.figure()
    plt.imshow(data, aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(agg_df)), agg_df["business_id"].values)
    plt.xticks(range(len(aspects)), aspects, rotation=45)
    plt.title("Aspect Positive Rates per Business")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aspect_heatmap.png"), dpi=160)
