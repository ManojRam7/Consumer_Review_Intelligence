"""Credible vs inconsistent reviews.

A review is marked **inconsistent** when at least one of these holds:

* **Contradiction** - the star rating and the wording disagree strongly: 4-5 stars with a
  VADER compound score <= -0.5, or 1-2 stars with a compound score >= +0.5
  (for example "Terrible book, waste of time, but 5 stars for quick delivery").
* **Too short to judge** - fewer than 3 words ("ok", "Good").
* **Generic text** - the same text posted for three or more different books
  ("Great book!", "Loved it"), which says nothing about the book itself.

Everything else is **credible**. Helpful votes are reported for both groups rather than
used as a rule, because most reviews have none.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from . import config as C


def flag_credibility(df: pd.DataFrame, threshold: float = C.CONTRADICTION_THRESHOLD,
                     min_words: int = C.MIN_WORDS_CREDIBLE) -> pd.DataFrame:
    """Expects `rating`, `text`, `n_words` and `vader_compound` columns."""
    out = df.copy()
    comp = out["vader_compound"]
    contradiction = ((out["rating"] >= 4) & (comp <= -threshold)) | ((out["rating"] <= 2) & (comp >= threshold))
    too_short = out["n_words"] < min_words

    if "parent_asin" in out.columns:
        norm = out["text"].str.lower().str.strip()
        books_per_text = out.groupby(norm)["parent_asin"].transform("nunique")
        generic = (books_per_text >= 3) & ~too_short
    else:
        generic = pd.Series(False, index=out.index)

    out["flag_contradiction"] = contradiction
    out["flag_too_short"] = too_short
    out["flag_generic"] = generic
    out["credibility"] = np.where(contradiction | too_short | generic, "Inconsistent", "Credible")
    out["inconsistency_reason"] = np.select(
        [contradiction, generic, too_short],
        ["rating contradicts text", "generic repeated text", "too short"], default="")
    return out


def credibility_summary(df: pd.DataFrame) -> dict:
    """Counts, sentiment mix per group (pd.crosstab), and helpful votes per group."""
    counts = df["credibility"].value_counts()
    mix = pd.crosstab(df["credibility"], df["sentiment"], normalize="index").reindex(columns=C.LABELS, fill_value=0)
    raw = pd.crosstab(df["credibility"], df["sentiment"]).reindex(columns=C.LABELS, fill_value=0)
    reasons = df.loc[df["credibility"] == "Inconsistent", "inconsistency_reason"].value_counts()
    summary = {
        "counts": {k: int(v) for k, v in counts.items()},
        "inconsistent_share": float((df["credibility"] == "Inconsistent").mean()),
        "reasons": {k: int(v) for k, v in reasons.items()},
        "sentiment_counts": {g: {k: int(v) for k, v in row.items()} for g, row in raw.iterrows()},
        "sentiment_mix": {g: {k: round(float(v), 4) for k, v in row.items()} for g, row in mix.iterrows()},
    }
    if "helpful_vote" in df.columns:
        hv = df.groupby("credibility")["helpful_vote"]
        summary["helpful_votes"] = {g: {"mean": round(float(s.mean()), 2), "median": float(s.median()),
                                        "share_with_votes": round(float((s > 0).mean()), 4)}
                                    for g, s in hv}
    return summary


def contradiction_examples(df: pd.DataFrame, n: int = 6, max_chars: int = 160) -> pd.DataFrame:
    """Most-voted contradictory reviews, with the text shortened."""
    ex = df[df["flag_contradiction"]].copy()
    sort_col = "helpful_vote" if "helpful_vote" in ex.columns else "vader_compound"
    ex = ex.sort_values(sort_col, ascending=False)
    text = ex["text"].astype(str).map(html.unescape).str.replace(r"<[^>]+>", " ", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    ex["excerpt"] = text.where(text.str.len() <= max_chars, text.str.slice(0, max_chars).str.rstrip() + "...")
    ex = ex.drop_duplicates("excerpt").head(n)
    cols = ["rating", "vader_compound", "excerpt"] + (["helpful_vote"] if "helpful_vote" in ex.columns else [])
    return ex[cols].round({"vader_compound": 3})
