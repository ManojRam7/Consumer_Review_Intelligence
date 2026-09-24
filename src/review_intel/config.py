"""Paths, data source and experiment settings in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "books_reviews.jsonl"
PROCESSED_PATH = DATA_DIR / "processed" / "reviews_clean.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
README_PATH = ROOT / "README.md"

# Amazon Reviews 2023 (McAuley Lab, UCSD), Books category. The file is streamed and
# only the first N_REVIEWS lines are kept, matching dataset["full"][:10000] in the dissertation.
HF_BOOKS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/Books.jsonl"
)
N_REVIEWS = 10_000

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80/20 split, stratified by sentiment
CV_FOLDS = 3              # 3-fold grid search, as in the dissertation

LABELS = ["negative", "neutral", "positive"]

# Vectoriser settings shared by Bag-of-Words and TF-IDF so the comparison is like for like.
VECTORIZER_PARAMS = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20_000)

# Rating vs text contradiction threshold on the VADER compound score (-1 to +1).
CONTRADICTION_THRESHOLD = 0.5
MIN_WORDS_CREDIBLE = 3

for _d in (RAW_PATH.parent, PROCESSED_PATH.parent, FIG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
