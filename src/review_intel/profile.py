"""Data profile of the raw reviews: shape, completeness, duplicates, distributions and flags.

Runs on the raw data (before cleaning) so that quality issues are visible, and writes
reports/data_profile.md and reports/data_profile.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C
from .preprocess import rating_to_sentiment


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _describe(s: pd.Series) -> dict:
    s = s.dropna()
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"min": float(s.min()), "p5": float(q[0.05]), "p25": float(q[0.25]),
            "median": float(q[0.5]), "mean": float(s.mean()), "p75": float(q[0.75]),
            "p95": float(q[0.95]), "p99": float(q[0.99]), "max": float(s.max())}


def profile_reviews(df: pd.DataFrame) -> dict:
    n = len(df)
    text = df["text"].fillna("").astype(str)
    words = text.str.split().str.len()
    sentiment = df["rating"].map(rating_to_sentiment)

    prof: dict = {"rows": n, "columns": list(df.columns)}
    prof["null_rate"] = {c: float(df[c].isna().mean()) for c in df.columns}
    prof["empty_text"] = int((text.str.strip() == "").sum())

    # Duplicates: same reviewer + product, and identical text posted more than once
    if {"user_id", "parent_asin"} <= set(df.columns):
        prof["duplicate_reviewer_product"] = int(df.duplicated(["user_id", "parent_asin"]).sum())
    dup_text = text[text.str.len() > 0].duplicated(keep=False)
    prof["duplicate_text_rows"] = int(dup_text.sum())

    prof["rating_counts"] = {str(int(k)): int(v) for k, v in df["rating"].value_counts().sort_index().items()}
    counts = sentiment.value_counts()
    prof["sentiment_counts"] = {k: int(counts.get(k, 0)) for k in C.LABELS}
    prof["imbalance_ratio"] = float(counts.max() / max(counts.min(), 1))

    prof["words_per_review"] = _describe(words)
    prof["chars_per_review"] = _describe(text.str.len())
    prof["very_short_reviews"] = int((words < C.MIN_WORDS_CREDIBLE).sum())

    if "helpful_vote" in df.columns:
        hv = df["helpful_vote"]
        prof["helpful_vote"] = _describe(hv)
        prof["helpful_vote_zero_share"] = float((hv == 0).mean())
    if "verified_purchase" in df.columns:
        prof["verified_share"] = float(df["verified_purchase"].astype(float).mean())
    if "review_date" in df.columns:
        d = df["review_date"].dropna()
        prof["date_min"] = str(d.min().date()) if len(d) else None
        prof["date_max"] = str(d.max().date()) if len(d) else None
        prof["reviews_per_year"] = {str(k): int(v) for k, v in d.dt.year.value_counts().sort_index().items()}
    if "parent_asin" in df.columns:
        per_book = df["parent_asin"].value_counts()
        prof["distinct_books"] = int(per_book.size)
        prof["top10_books_share"] = float(per_book.head(10).sum() / n) if n else 0.0

    prof["flags"] = _flags(prof)
    return prof


def _flags(p: dict) -> list[str]:
    flags = []
    for col, rate in p["null_rate"].items():
        if rate > 0.20:
            flags.append(f"ALERT: `{col}` is {_pct(rate)} null")
        elif rate > 0.05:
            flags.append(f"WARN: `{col}` is {_pct(rate)} null")
    if p["empty_text"]:
        flags.append(f"{p['empty_text']} reviews have empty text and are dropped")
    if p.get("duplicate_text_rows"):
        flags.append(f"{p['duplicate_text_rows']} rows share identical text with another review "
                     "(short stock phrases or copy-pasted reviews)")
    if p["imbalance_ratio"] > 3:
        flags.append(f"Class imbalance: largest class is {p['imbalance_ratio']:.1f}x the smallest; "
                     "use stratified splits, class weights and macro-averaged metrics")
    if p.get("helpful_vote_zero_share", 0) > 0.5:
        flags.append(f"{_pct(p['helpful_vote_zero_share'])} of reviews have no helpful votes, so votes are "
                     "a weak credibility signal on their own")
    if p["very_short_reviews"]:
        flags.append(f"{p['very_short_reviews']} reviews have fewer than {C.MIN_WORDS_CREDIBLE} words")
    return flags


def write_profile(prof: dict, out_dir: Path = C.REPORTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_profile.json").write_text(json.dumps(prof, indent=2, default=str))

    n = prof["rows"]
    s = prof["sentiment_counts"]
    w = prof["words_per_review"]
    lines = [
        "# Data profile: Amazon Books reviews",
        "",
        f"- **Rows:** {n:,} (one row per review)",
        f"- **Columns:** {', '.join(f'`{c}`' for c in prof['columns'])}",
    ]
    if prof.get("date_min"):
        lines.append(f"- **Review dates:** {prof['date_min']} to {prof['date_max']}")
    if prof.get("distinct_books"):
        lines.append(f"- **Distinct books:** {prof['distinct_books']:,} "
                     f"(top 10 books hold {_pct(prof['top10_books_share'])} of reviews)")
    lines += [
        "",
        "## Labels",
        "",
        "| Sentiment | Rule | Reviews | Share |",
        "|---|---|---|---|",
    ]
    rules = {"positive": "4-5 stars", "neutral": "3 stars", "negative": "1-2 stars"}
    for k in reversed(C.LABELS):
        lines.append(f"| {k} | {rules[k]} | {s[k]:,} | {_pct(s[k] / n) if n else '-'} |")
    lines += [
        "",
        "| Stars | " + " | ".join(prof["rating_counts"].keys()) + " |",
        "|---|" + "---|" * len(prof["rating_counts"]),
        "| Reviews | " + " | ".join(f"{v:,}" for v in prof["rating_counts"].values()) + " |",
        "",
        "## Text length",
        "",
        "| Words per review | min | p5 | median | mean | p95 | max |",
        "|---|---|---|---|---|---|---|",
        f"| | {w['min']:.0f} | {w['p5']:.0f} | {w['median']:.0f} | {w['mean']:.1f} | {w['p95']:.0f} | {w['max']:.0f} |",
        "",
        "## Completeness and duplicates",
        "",
        "| Column | Null rate |",
        "|---|---|",
    ]
    lines += [f"| `{c}` | {_pct(r)} |" for c, r in prof["null_rate"].items()]
    lines += ["", f"- Empty review text: {prof['empty_text']:,}",
              f"- Rows with text identical to another review: {prof['duplicate_text_rows']:,}"]
    if "duplicate_reviewer_product" in prof:
        lines.append(f"- Same reviewer reviewing the same book twice: {prof['duplicate_reviewer_product']:,}")
    if "helpful_vote" in prof:
        hv = prof["helpful_vote"]
        lines += ["", "## Helpful votes", "",
                  f"- {_pct(prof['helpful_vote_zero_share'])} of reviews have zero helpful votes",
                  f"- Median {hv['median']:.0f}, p95 {hv['p95']:.0f}, max {hv['max']:.0f}"]
    if "verified_share" in prof:
        lines.append(f"- Verified purchases: {_pct(prof['verified_share'])}")
    lines += ["", "## Flags", ""] + [f"- {f}" for f in prof["flags"] or ["No issues flagged"]]
    path = out_dir / "data_profile.md"
    path.write_text("\n".join(lines) + "\n")
    return path
