"""Text cleaning and sentiment labels.

Steps (in order): HTML unescape and tag removal, lowercase, URLs removed, emojis
converted to words, contractions expanded, slang expanded, punctuation and numbers
removed, stop words removed (negations kept), short tokens dropped, lemmatisation.
"""
from __future__ import annotations

import html
import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

SLANG = {
    "gr8": "great", "grt": "great", "luv": "love", "lov": "love", "u": "you", "ur": "your",
    "r": "are", "pls": "please", "plz": "please", "thx": "thanks", "thnx": "thanks",
    "ty": "thank you", "b4": "before", "bc": "because", "cuz": "because", "coz": "because",
    "idk": "i do not know", "imo": "in my opinion", "imho": "in my opinion", "tbh": "to be honest",
    "omg": "oh my god", "lol": "laughing", "lmao": "laughing", "awsm": "awesome", "fav": "favourite",
    "fave": "favourite", "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of",
    "sorta": "sort of", "dunno": "do not know", "w/": "with", "w/o": "without", "bday": "birthday",
    "yr": "year", "yrs": "years", "min": "minute", "mins": "minutes", "bk": "book", "bks": "books",
    "rec": "recommend", "recs": "recommendations", "meh": "mediocre", "ok": "okay", "k": "okay",
}

CONTRACTIONS = [
    (r"won['’]t", "will not"), (r"can['’]t", "can not"), (r"shan['’]t", "shall not"),
    (r"n['’]t\b", " not"), (r"['’]re\b", " are"), (r"['’]m\b", " am"), (r"['’]ll\b", " will"),
    (r"['’]ve\b", " have"), (r"['’]d\b", " would"), (r"['’]s\b", ""),
]

# Negations flip sentiment, so they stay even though standard stop-word lists include them.
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere",
             "without", "hardly", "barely"}

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")
_SLANG_TOKEN = re.compile(
    r"(?<![\w/.'’-])(" + "|".join(re.escape(k) for k in sorted(SLANG, key=len, reverse=True)) + r")(?![\w/.'’-])"
)


class TextCleaner:
    """Reusable cleaner; loads stop words and the lemmatiser once."""

    def __init__(self, lemmatise: bool = True):
        self.stop_words = self._load_stop_words() - NEGATIONS
        self.lemmatizer = self._load_lemmatizer() if lemmatise else None
        try:
            import emoji  # noqa: F401
            self._emoji = emoji
        except ImportError:
            self._emoji = None
            log.warning("`emoji` not installed; emojis will be dropped instead of converted to words")

    @staticmethod
    def _load_stop_words() -> set[str]:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except (ImportError, LookupError):
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            log.warning("NLTK stop words unavailable; using scikit-learn's English list")
            return set(ENGLISH_STOP_WORDS)

    @staticmethod
    def _load_lemmatizer():
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("books")          # raises LookupError if WordNet is missing
            return lem
        except (ImportError, LookupError):
            log.warning("WordNet unavailable; skipping lemmatisation")
            return None

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        t = html.unescape(text)
        t = _HTML_TAG.sub(" ", t)
        t = t.lower()
        t = _URL.sub(" ", t)
        if self._emoji is not None:
            t = self._emoji.demojize(t, delimiters=(" ", " ")).replace("_", " ")
        for pattern, repl in CONTRACTIONS:
            t = re.sub(pattern, repl, t)
        t = _SLANG_TOKEN.sub(lambda m: SLANG[m.group(1)], t)
        t = _NON_ALPHA.sub(" ", t)
        tokens = [w for w in _SPACES.split(t) if len(w) > 1 and w not in self.stop_words]
        if self.lemmatizer is not None:
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(tokens)


def ensure_nltk_resources(quiet: bool = True) -> None:
    """Download the NLTK corpora used here if they are missing (needs internet once)."""
    try:
        import nltk
    except ImportError:
        return
    for resource, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                           ("omw-1.4", "corpora/omw-1.4")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=quiet)
            except Exception as exc:  # offline: the cleaner falls back gracefully
                log.warning("Could not download NLTK resource %s: %s", resource, exc)


def rating_to_sentiment(rating: float) -> str | None:
    """Positive = 4-5 stars, neutral = 3, negative = 1-2 (the dissertation's labelling)."""
    if pd.isna(rating):
        return None
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def prepare(df: pd.DataFrame, cleaner: TextCleaner | None = None) -> pd.DataFrame:
    """Drop unusable rows, label sentiment, clean text and add simple text statistics."""
    cleaner = cleaner or TextCleaner()
    out = df.copy()
    before = len(out)
    out = out.dropna(subset=["rating", "text"])
    out = out[out["text"].str.strip().str.len() > 0]
    out = out.drop_duplicates(subset=["text", "rating", "parent_asin"] if "parent_asin" in out else ["text", "rating"])
    log.info("Dropped %d rows with missing text/rating or exact duplicates", before - len(out))

    out["sentiment"] = out["rating"].map(rating_to_sentiment)
    out["clean_text"] = out["text"].astype(str).map(cleaner.clean)
    out = out[out["clean_text"].str.len() > 0]
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].str.split().str.len()
    return out.reset_index(drop=True)
