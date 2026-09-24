"""Consumer Review Intelligence: end-to-end run.

    python run_pipeline.py                 # first 10,000 Amazon Books reviews, full grids
    python run_pipeline.py --quick         # smaller grids (linear SVM only), a few minutes
    python run_pipeline.py --input my.csv  # your own file with `rating` and `text` columns

Stages: download -> profile -> clean and label -> VADER + credibility -> exploratory
charts -> train and tune 6 model/feature pairs -> evaluate -> reports and README results.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from review_intel import config as C  # noqa: E402
from review_intel import credibility, data, insights, models, report  # noqa: E402
from review_intel.features import vader_scores  # noqa: E402
from review_intel.preprocess import TextCleaner, ensure_nltk_resources, prepare  # noqa: E402
from review_intel.profile import profile_reviews, write_profile  # noqa: E402

log = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=C.N_REVIEWS, help="number of reviews (default 10,000)")
    p.add_argument("--input", type=Path, help="local JSONL/CSV instead of downloading")
    p.add_argument("--quick", action="store_true", help="reduced hyperparameter grids")
    p.add_argument("--oversample", action="store_true",
                   help="random oversampling of minority classes instead of class weights (needs imbalanced-learn)")
    p.add_argument("--jobs", type=int, default=-1, help="parallel jobs for grid search (default: all cores)")
    p.add_argument("--no-readme", action="store_true", help="do not rewrite the README results block")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    t0 = time.time()

    # 1. Data
    path = args.input or data.download_reviews(n=args.n)
    raw = data.read_reviews(path, n=args.n)
    log.info("Loaded %s reviews from %s", f"{len(raw):,}", path)

    # 2. Profile (before anything is dropped), then remove reviewer IDs
    prof = profile_reviews(raw)
    log.info("Data profile -> %s", write_profile(prof))
    raw = data.anonymise(raw)

    # 3. Clean text and label sentiment from stars
    ensure_nltk_resources()
    df = prepare(raw, TextCleaner())
    counts = df["sentiment"].value_counts().to_dict()
    log.info("After cleaning: %s reviews | %s", f"{len(df):,}", counts)

    # 4. Lexicon scores and credibility flags
    df = df.join(vader_scores(df["text"]))
    df = credibility.flag_credibility(df)
    cred = credibility.credibility_summary(df)
    log.info("Inconsistent reviews: %.1f%% %s", cred["inconsistent_share"] * 100, cred["reasons"])
    df.drop(columns=["text"]).to_csv(C.PROCESSED_PATH, index=False)

    # 5. Exploratory charts and top terms
    eda = insights.build_all(df)

    # 6. Models: stratified 80/20 split, grid search on the training set only
    train, test = train_test_split(df, test_size=C.TEST_SIZE, stratify=df["sentiment"],
                                   random_state=C.RANDOM_STATE)
    results = models.run_experiments(train["clean_text"], train["sentiment"], test["clean_text"],
                                     test["sentiment"], quick=args.quick, oversample=args.oversample,
                                     n_jobs=args.jobs)

    # 7. Reports
    best = max((r for r in results if r.model != "Majority class"), key=lambda r: r.metrics["f1_macro"])
    joblib.dump(best.estimator, C.MODELS_DIR / "best_model.joblib")
    report.model_comparison_chart(results)
    report.confusion_grid(results)
    report.random_forest_charts(results)

    summary = {"n_reviews": len(df), "n_train": len(train), "n_test": len(test),
               "class_counts": {k: int(counts.get(k, 0)) for k in C.LABELS},
               "grid": "quick" if args.quick else "full",
               "imbalance_handling": "random oversampling" if args.oversample else "class weights",
               "best_model": best.name, "credibility": cred,
               "vocabulary_size": eda["vocabulary_size"]}
    report.save_metrics(results, summary)
    examples = credibility.contradiction_examples(df)
    report.write_results_md(results, summary, examples, eda)
    if not args.no_readme and report.update_readme(report.results_markdown(results, summary)):
        log.info("README results section updated")

    log.info("Best: %s (macro F1 %.3f). Done in %.1f min.", best.name, best.metrics["f1_macro"],
             (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
