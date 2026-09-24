import pandas as pd

from review_intel.credibility import credibility_summary, flag_credibility


def _frame():
    return pd.DataFrame({
        "rating": [5, 5, 1, 4, 5, 5, 5],
        "text": ["Terrible book, waste of time, but five stars for quick delivery",
                 "Wonderful characters and a moving story",
                 "Brilliant, loved every page, best book this year",
                 "Good",
                 "Great gift for any reader", "Great gift for any reader", "Great gift for any reader"],
        "n_words": [11, 6, 8, 1, 5, 5, 5],
        "vader_compound": [-0.75, 0.80, 0.90, 0.44, 0.62, 0.62, 0.62],
        "parent_asin": ["A", "B", "C", "D", "E", "F", "G"],
        "sentiment": ["positive", "positive", "negative", "positive", "positive", "positive", "positive"],
        "helpful_vote": [2, 25, 1, 0, 0, 0, 0],
    })


def test_rules():
    out = flag_credibility(_frame())
    assert list(out["credibility"]) == ["Inconsistent", "Credible", "Inconsistent", "Inconsistent",
                                        "Inconsistent", "Inconsistent", "Inconsistent"]
    assert out.loc[0, "inconsistency_reason"] == "rating contradicts text"
    assert out.loc[2, "inconsistency_reason"] == "rating contradicts text"
    assert out.loc[3, "inconsistency_reason"] == "too short"
    assert out.loc[4, "inconsistency_reason"] == "generic repeated text"


def test_summary_crosstab():
    s = credibility_summary(flag_credibility(_frame()))
    assert s["counts"] == {"Inconsistent": 6, "Credible": 1}
    assert abs(sum(s["sentiment_mix"]["Inconsistent"].values()) - 1) < 1e-9
    assert s["helpful_votes"]["Credible"]["mean"] == 25
