"""Text representations (Bag-of-Words, TF-IDF) and lexicon sentiment scores (VADER)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config as C


def make_vectorizer(kind: str):
    """'bow' -> word and bigram counts; 'tfidf' -> the same terms weighted by TF-IDF."""
    if kind == "bow":
        return CountVectorizer(**C.VECTORIZER_PARAMS)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **C.VECTORIZER_PARAMS)
    raise ValueError(f"Unknown vectoriser: {kind}")


FEATURE_NAMES = {"bow": "BoW", "tfidf": "TF-IDF"}


def vader_scores(texts: pd.Series) -> pd.DataFrame:
    """VADER polarity on the raw review text (it relies on punctuation and capitals)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    scores = [analyser.polarity_scores(str(t)) for t in texts]
    out = pd.DataFrame(scores, index=texts.index)
    return out.rename(columns={"compound": "vader_compound", "pos": "vader_pos",
                               "neu": "vader_neu", "neg": "vader_neg"})


def vader_label(compound: pd.Series) -> pd.Series:
    """Standard VADER cut-offs: >= 0.05 positive, <= -0.05 negative, otherwise neutral."""
    return pd.Series(np.select([compound >= 0.05, compound <= -0.05], ["positive", "negative"], "neutral"),
                     index=compound.index)


def top_terms(texts: pd.Series, kind: str, n: int = 20) -> pd.DataFrame:
    """Most frequent (BoW) or highest total-weight (TF-IDF) unigrams across the corpus."""
    params = {**C.VECTORIZER_PARAMS, "ngram_range": (1, 1)}
    vec = CountVectorizer(**params) if kind == "bow" else TfidfVectorizer(sublinear_tf=True, **params)
    X = vec.fit_transform(texts)
    totals = np.asarray(X.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = totals.argsort()[::-1][:n]
    col = "count" if kind == "bow" else "tfidf_weight"
    return pd.DataFrame({"term": terms[order], col: totals[order].round(3)})


def top_ngrams(texts: pd.Series, n: int = 15, ngram: int = 2) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=(ngram, ngram), min_df=2)
    X = vec.fit_transform(texts)
    totals = np.asarray(X.sum(axis=0)).ravel()
    order = totals.argsort()[::-1][:n]
    return pd.DataFrame({"ngram": vec.get_feature_names_out()[order], "count": totals[order]})
