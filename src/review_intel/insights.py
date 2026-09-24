"""Exploratory charts: class balance, review length, VADER vs stars, word clouds, top terms,
sentiment over time and the credible vs inconsistent breakdown."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import top_ngrams, top_terms

log = logging.getLogger(__name__)

COLORS = {"positive": "#0d9488", "neutral": "#94a3b8", "negative": "#e11d48"}
INK = "#0f172a"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
                     "font.size": 10})


def _save(fig, name: str, out: Path = C.FIG_DIR) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    stars = df["rating"].value_counts().sort_index()
    axes[0].bar(stars.index.astype(int).astype(str), stars.values, color="#6366f1")
    axes[0].set_title("Reviews by star rating")
    axes[0].set_xlabel("Stars")
    sent = df["sentiment"].value_counts().reindex(C.LABELS).fillna(0).astype(int)
    axes[1].bar(sent.index, sent.values, color=[COLORS[k] for k in sent.index])
    for i, v in enumerate(sent.values):
        axes[1].text(i, v, f"{v / sent.sum():.0%}", ha="center", va="bottom")
    axes[1].set_title("Sentiment labels (from stars)")
    return _save(fig, "class_balance.png")


def length_by_sentiment(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    data = [df.loc[df["sentiment"] == k, "n_words"].clip(lower=1) for k in C.LABELS]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticks(range(1, len(C.LABELS) + 1), C.LABELS)
    for patch, k in zip(bp["boxes"], C.LABELS):
        patch.set_facecolor(COLORS[k])
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_ylabel("Words per review (log scale)")
    ax.set_title("Review length by sentiment")
    return _save(fig, "length_by_sentiment.png")


def vader_by_rating(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    stars = sorted(df["rating"].dropna().unique())
    ax.boxplot([df.loc[df["rating"] == s, "vader_compound"] for s in stars], showfliers=False)
    ax.set_xticks(range(1, len(stars) + 1), [str(int(s)) for s in stars])
    ax.axhline(0, color="#cbd5e1", lw=1)
    ax.set_xlabel("Stars")
    ax.set_ylabel("VADER compound score")
    ax.set_title("Lexicon sentiment vs star rating")
    return _save(fig, "vader_by_rating.png")


def wordclouds(df: pd.DataFrame) -> list[Path]:
    try:
        from wordcloud import WordCloud
    except ImportError:
        log.warning("wordcloud not installed; skipping word clouds")
        return []
    paths = []
    groups = {k: df.loc[df["sentiment"] == k, "clean_text"] for k in C.LABELS}
    groups["all"] = df["clean_text"]
    cmaps = {"positive": "GnBu", "neutral": "Greys", "negative": "RdPu", "all": "viridis"}
    for name, texts in groups.items():
        text = " ".join(texts)
        if not text.strip():
            continue
        wc = WordCloud(width=1000, height=500, background_color="white", colormap=cmaps[name],
                       max_words=120, collocations=False, random_state=C.RANDOM_STATE).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word cloud: {name} reviews" if name != "all" else "Word cloud: all reviews")
        paths.append(_save(fig, f"wordcloud_{name}.png"))
    return paths


def term_charts(df: pd.DataFrame) -> dict:
    bow = top_terms(df["clean_text"], "bow", n=20)
    tfidf = top_terms(df["clean_text"], "tfidf", n=20)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    top10 = bow.head(10).iloc[::-1]
    ax.barh(top10["term"], top10["count"], color="#6366f1")
    ax.set_title("Top 10 words (Bag-of-Words counts)")
    _save(fig, "top10_bow_words.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    t = tfidf.iloc[::-1]
    ax.barh(t["term"], t["tfidf_weight"], color="#7c3aed")
    ax.set_title("Top 20 words by total TF-IDF weight")
    _save(fig, "top20_tfidf_words.png")

    # Rank-frequency plot: a handful of words carry most of the mass (Zipf's law)
    from sklearn.feature_extraction.text import CountVectorizer
    counts = np.asarray(CountVectorizer().fit_transform(df["clean_text"]).sum(axis=0)).ravel()
    counts = np.sort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(np.arange(1, len(counts) + 1), counts, color="#0d9488")
    ax.set_xlabel("Word rank")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Word frequency distribution ({len(counts):,} unique words)")
    _save(fig, "word_frequency_distribution.png")

    bigrams = {k: top_ngrams(df.loc[df["sentiment"] == k, "clean_text"], n=10)
               for k in C.LABELS if (df["sentiment"] == k).sum() > 20}
    return {"top_bow": bow, "top_tfidf": tfidf, "top_bigrams": bigrams, "vocabulary_size": int(len(counts))}


def sentiment_over_time(df: pd.DataFrame) -> Path | None:
    if "review_date" not in df.columns or df["review_date"].isna().all():
        return None
    year = pd.to_datetime(df["review_date"]).dt.year
    mix = pd.crosstab(year, df["sentiment"], normalize="index").reindex(columns=C.LABELS, fill_value=0)
    volume = year.value_counts().sort_index()
    mix = mix.loc[volume[volume >= 30].index]          # skip years with too few reviews
    if mix.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(mix))
    for k in C.LABELS:
        ax.bar(mix.index.astype(str), mix[k], bottom=bottom, color=COLORS[k], label=k)
        bottom += mix[k].values
    ax.set_ylabel("Share of reviews")
    ax.set_title("Sentiment mix by year (years with 30+ reviews)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, "sentiment_by_year.png")


def credibility_pies(df: pd.DataFrame) -> Path:
    tab = pd.crosstab(df["credibility"], df["sentiment"]).reindex(columns=C.LABELS, fill_value=0)
    groups = [g for g in ("Credible", "Inconsistent") if g in tab.index]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.4))
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        row = tab.loc[g]
        row = row[row > 0]
        ax.pie(row.values, labels=row.index, autopct="%1.0f%%", startangle=90,
               colors=[COLORS[k] for k in row.index], wedgeprops={"edgecolor": "white"})
        ax.set_title(f"{g} reviews (n={int(tab.loc[g].sum()):,})")
    return _save(fig, "credibility_sentiment_pies.png")


def build_all(df: pd.DataFrame) -> dict:
    out = {"figures": []}
    out["figures"] += [class_balance(df), length_by_sentiment(df), vader_by_rating(df)]
    out["figures"] += wordclouds(df)
    out.update(term_charts(df))
    p = sentiment_over_time(df)
    if p:
        out["figures"].append(p)
    out["figures"].append(credibility_pies(df))
    log.info("Wrote %d exploratory figures to %s", len(out["figures"]), C.FIG_DIR)
    return out
