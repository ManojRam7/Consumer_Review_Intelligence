"""Sentiment classifiers: SVM, Multinomial Naive Bayes and Random Forest, each on BoW and TF-IDF.

Every model is a Pipeline (vectoriser -> classifier), so the vocabulary and IDF weights are
learned inside each cross-validation fold and never see the test set. Hyperparameters are
tuned with GridSearchCV (3-fold, stratified) and selected on macro F1, because positive
reviews dominate and accuracy alone rewards always predicting "positive".
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C
from .features import FEATURE_NAMES, make_vectorizer

log = logging.getLogger(__name__)

# 4 x 3 x 2 x 2 = 48 candidates x 3 folds = 144 fits: the grid used in the dissertation.
SVM_GRID = {"clf__C": [0.1, 1, 10, 100], "clf__kernel": ["linear", "rbf", "poly"],
            "clf__gamma": ["scale", "auto"], "clf__degree": [2, 3]}
SVM_GRID_QUICK = {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear"]}

NB_GRID = {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]}
NB_GRID_QUICK = {"clf__alpha": [0.1, 1.0]}

RF_GRID = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 60],
           "clf__min_samples_leaf": [1, 2]}
RF_GRID_QUICK = {"clf__n_estimators": [200], "clf__min_samples_leaf": [1]}


def _classifiers(oversample: bool) -> dict:
    # With oversampling the classes are already balanced, so no extra weighting.
    weight = None if oversample else "balanced"
    return {
        "SVM": SVC(class_weight=weight, random_state=C.RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(class_weight=None if oversample else "balanced_subsample",
                                                random_state=C.RANDOM_STATE, n_jobs=1),
    }


def _grids(quick: bool) -> dict:
    if quick:
        return {"SVM": SVM_GRID_QUICK, "Naive Bayes": NB_GRID_QUICK, "Random Forest": RF_GRID_QUICK}
    return {"SVM": SVM_GRID, "Naive Bayes": NB_GRID, "Random Forest": RF_GRID}


def build_pipeline(features: str, clf, oversample: bool = False):
    if oversample:
        try:
            from imblearn.over_sampling import RandomOverSampler
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:
            raise ImportError("--oversample needs imbalanced-learn: pip install imbalanced-learn") from exc
        return ImbPipeline([("vec", make_vectorizer(features)),
                            ("resample", RandomOverSampler(random_state=C.RANDOM_STATE)),
                            ("clf", clf)])
    return Pipeline([("vec", make_vectorizer(features)), ("clf", clf)])


@dataclass
class Result:
    model: str
    features: str
    metrics: dict
    best_params: dict = field(default_factory=dict)
    cv_f1_macro: float | None = None
    n_candidates: int = 0
    fit_seconds: float = 0.0
    confusion: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    estimator: object = None

    @property
    def name(self) -> str:
        return f"{self.model} + {FEATURE_NAMES.get(self.features, self.features)}"

    def to_dict(self) -> dict:
        return {"model": self.model, "features": FEATURE_NAMES.get(self.features, self.features),
                "metrics": self.metrics, "best_params": self.best_params,
                "cv_f1_macro": self.cv_f1_macro, "n_candidates": self.n_candidates,
                "fit_seconds": round(self.fit_seconds, 1), "confusion_matrix": self.confusion,
                "classification_report": self.report}


def evaluate(y_true, y_pred) -> tuple[dict, list, dict]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=C.LABELS,
                                                 average="macro", zero_division=0)
    metrics = {"accuracy": round(accuracy_score(y_true, y_pred), 4),
               "precision_macro": round(p, 4), "recall_macro": round(r, 4), "f1_macro": round(f, 4),
               "f1_weighted": round(f1_score(y_true, y_pred, labels=C.LABELS, average="weighted",
                                             zero_division=0), 4)}
    cm = confusion_matrix(y_true, y_pred, labels=C.LABELS).tolist()
    report = classification_report(y_true, y_pred, labels=C.LABELS, output_dict=True, zero_division=0)
    return metrics, cm, report


def run_baseline(X_train, y_train, X_test, y_test) -> Result:
    pipe = Pipeline([("vec", make_vectorizer("bow")), ("clf", DummyClassifier(strategy="most_frequent"))])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    metrics, cm, report = evaluate(y_test, pipe.predict(X_test))
    return Result("Majority class", "bow", metrics, fit_seconds=time.time() - t0,
                  confusion=cm, report=report, estimator=pipe)


def run_experiments(X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series,
                    quick: bool = False, oversample: bool = False, n_jobs: int = -1) -> list[Result]:
    """Grid-search every classifier on both feature sets and score it on the held-out test set."""
    results = [run_baseline(X_train, y_train, X_test, y_test)]
    classifiers, grids = _classifiers(oversample), _grids(quick)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    for model_name, clf in classifiers.items():
        for features in ("bow", "tfidf"):
            pipe = build_pipeline(features, clf, oversample)
            search = GridSearchCV(pipe, grids[model_name], scoring="f1_macro", cv=cv,
                                  n_jobs=n_jobs, refit=True, verbose=1)
            fit_params = {}
            if model_name == "Naive Bayes" and not oversample:
                # MultinomialNB has no class_weight; balanced sample weights do the same job
                fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)
            log.info("Tuning %s + %s (%d candidates)", model_name, FEATURE_NAMES[features],
                     int(np.prod([len(v) for v in grids[model_name].values()])))
            t0 = time.time()
            search.fit(X_train, y_train, **fit_params)
            elapsed = time.time() - t0

            metrics, cm, report = evaluate(y_test, search.predict(X_test))
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
            res = Result(model_name, features, metrics, best_params=params,
                         cv_f1_macro=round(float(search.best_score_), 4),
                         n_candidates=len(search.cv_results_["params"]), fit_seconds=elapsed,
                         confusion=cm, report=report, estimator=search.best_estimator_)
            log.info("%s: accuracy %.3f | macro F1 %.3f | best %s", res.name,
                     metrics["accuracy"], metrics["f1_macro"], params)
            results.append(res)
    return results


def results_table(results: list[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"Model": r.model, "Features": FEATURE_NAMES.get(r.features, r.features)
                     if r.model != "Majority class" else "-", **r.metrics,
                     "Best parameters": ", ".join(f"{k}={v}" for k, v in r.best_params.items()) or "-"})
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
