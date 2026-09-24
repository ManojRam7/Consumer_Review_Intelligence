import pandas as pd

from review_intel.preprocess import TextCleaner, prepare, rating_to_sentiment

cleaner = TextCleaner(lemmatise=False)


def test_rating_labels():
    assert [rating_to_sentiment(r) for r in (1, 2, 3, 4, 5)] == \
        ["negative", "negative", "neutral", "positive", "positive"]
    assert rating_to_sentiment(float("nan")) is None


def test_html_urls_numbers_and_punctuation_removed():
    out = cleaner.clean("Loved it!<br />See https://example.com for 2 more&amp;more")
    assert "br" not in out.split()
    assert "http" not in out and "example" not in out
    assert "2" not in out and "!" not in out


def test_slang_and_contractions_expanded():
    out = cleaner.clean("This book is gr8, I didn't expect it")
    assert "great" in out.split()
    assert "not" in out.split()          # negation kept even though it is a stop word


def test_stop_words_removed_but_negations_kept():
    tokens = cleaner.clean("The story was not good and the ending was never explained").split()
    assert "the" not in tokens and "was" not in tokens
    assert "not" in tokens and "never" in tokens


def test_prepare_drops_empty_and_duplicate_rows():
    df = pd.DataFrame({
        "rating": [5, 5, 1, None, 3],
        "text": ["Great read", "Great read", "Awful", "Missing rating", "   "],
        "parent_asin": ["A", "A", "B", "C", "D"],
    })
    df["text"] = df["text"].astype("string")
    out = prepare(df, cleaner)
    assert len(out) == 2
    assert set(out["sentiment"]) == {"positive", "negative"}
    assert {"clean_text", "n_words", "n_chars"} <= set(out.columns)
