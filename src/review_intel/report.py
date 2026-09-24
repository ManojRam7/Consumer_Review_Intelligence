"""Model charts, metrics files and the results section of the README."""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURE_NAMES
from .insights import _save
from .models import Result, results_table

log = logging.getLogger(__name__)

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def model_comparison_chart(results: list[Result]) -> Path:
    tab = results_table(results)
    tab = tab[tab["Model"] != "Majority class"]
    labels = [f"{m}\n{f}" for m, f in zip(tab["Model"], tab["Features"])]
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - 0.2, tab["f1_macro"], 0.4, label="Macro F1", color="#4f46e5")
    ax.bar(x + 0.2, tab["accuracy"], 0.4, label="Accuracy", color="#0d9488")
    base = next((r for r in results if r.model == "Majority class"), None)
    if base:
        ax.axhline(base.metrics["accuracy"], color="#94a3b8", ls="--", lw=1,
                   label=f"Majority-class accuracy ({base.metrics['accuracy']:.2f})")
    for i, (f1, acc) in enumerate(zip(tab["f1_macro"], tab["accuracy"])):
        ax.text(i - 0.2, f1 + 0.01, f"{f1:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("Test-set performance by model and features (sorted by macro F1)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    return _save(fig, "model_comparison.png")


def confusion_grid(results: list[Result]) -> Path:
    models = [r for r in results if r.model != "Majority class"]
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, models):
        cm = np.array(r.confusion)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "#0f172a")
        ax.set_xticks(range(len(C.LABELS)), C.LABELS, fontsize=8)
        ax.set_yticks(range(len(C.LABELS)), C.LABELS, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{r.name}\nmacro F1 {r.metrics['f1_macro']:.2f}", fontsize=10)
    for ax in axes[len(models):]:
        ax.axis("off")
    return _save(fig, "confusion_matrices.png")


def random_forest_charts(results: list[Result]) -> list[Path]:
    """Feature importance and the first tree of the TF-IDF random forest."""
    rf = next((r for r in results if r.model == "Random Forest" and r.features == "tfidf"), None)
    if rf is None or rf.estimator is None:
        return []
    vec, clf = rf.estimator.named_steps["vec"], rf.estimator.named_steps["clf"]
    names = vec.get_feature_names_out()
    imp = pd.Series(clf.feature_importances_, index=names).nlargest(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(imp.index, imp.values, color="#7c3aed")
    ax.set_title("Random Forest (TF-IDF): top 20 features")
    paths = [_save(fig, "rf_feature_importance.png")]

    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(clf.estimators_[0], max_depth=3, feature_names=list(names), class_names=list(clf.classes_),
              filled=True, rounded=True, fontsize=7, ax=ax, impurity=False)
    ax.set_title("First tree of the Random Forest (top 3 levels)")
    paths.append(_save(fig, "rf_first_tree.png"))
    return paths


def save_metrics(results: list[Result], extra: dict) -> Path:
    payload = {"generated": date.today().isoformat(), **extra,
               "results": [r.to_dict() for r in results]}
    path = C.REPORTS_DIR / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def results_markdown(results: list[Result], summary: dict, prefix: str = "reports/") -> str:
    """The block written between the RESULTS markers in README.md."""
    tab = results_table(results)
    best = tab[tab["Model"] != "Majority class"].iloc[0]
    base = tab[tab["Model"] == "Majority class"].iloc[0]
    cred = summary["credibility"]
    n_train, n_test = summary["n_train"], summary["n_test"]
    counts = summary["class_counts"]
    total = sum(counts.values())

    lines = [
        f"_Latest run: {date.today():%d %B %Y}. {total:,} reviews after cleaning "
        f"({n_train:,} train / {n_test:,} test, stratified). Class mix: "
        + ", ".join(f"{k} {counts[k] / total:.0%}" for k in reversed(C.LABELS)) + "._",
        "",
        "| Model | Features | Accuracy | Macro precision | Macro recall | Macro F1 | Best parameters |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        bold = "**" if (r["Model"] == best["Model"] and r["Features"] == best["Features"]) else ""
        lines.append(f"| {bold}{r['Model']}{bold} | {r['Features']} | {r['accuracy']:.3f} | "
                     f"{r['precision_macro']:.3f} | {r['recall_macro']:.3f} | {bold}{r['f1_macro']:.3f}{bold} | "
                     f"{r['Best parameters']} |")
    by_feat = tab[tab["Model"] != "Majority class"].groupby("Features")["f1_macro"].mean()
    lines += [
        "",
        f"- **Best model:** {best['Model']} + {best['Features']}, macro F1 {best['f1_macro']:.3f} and "
        f"accuracy {best['accuracy']:.3f}, against {base['accuracy']:.3f} accuracy (macro F1 "
        f"{base['f1_macro']:.3f}) for always predicting the majority class.",
    ]
    if {"BoW", "TF-IDF"} <= set(by_feat.index):
        lines.append(f"- **Features:** average macro F1 across the three models is {by_feat['TF-IDF']:.3f} "
                     f"with TF-IDF and {by_feat['BoW']:.3f} with Bag-of-Words.")
    lines += [
        f"- **Credibility:** {cred['inconsistent_share']:.1%} of reviews are flagged inconsistent "
        + "(" + ", ".join(f"{v:,} {k}" for k, v in cred["reasons"].items()) + ").",
        "",
        "Full per-class reports, confusion matrices and tuned parameters: "
        f"[`reports/metrics.json`]({prefix}metrics.json) and [`reports/results.md`]({prefix}results.md).",
    ]
    return "\n".join(lines)


def write_results_md(results: list[Result], summary: dict, examples: pd.DataFrame, top_terms: dict) -> Path:
    cred = summary["credibility"]
    lines = ["# Results", "", results_markdown(results, summary, prefix=""), "", "## Per-class F1 (test set)", "",
             "| Model | Features | negative | neutral | positive |", "|---|---|---|---|---|"]
    for r in results:
        rep = r.report
        feats = FEATURE_NAMES.get(r.features, r.features) if r.model != "Majority class" else "-"
        lines.append(f"| {r.model} | {feats} | " + " | ".join(f"{rep[k]['f1-score']:.3f}" for k in C.LABELS) + " |")

    lines += ["", "## Credible vs inconsistent reviews", "",
              "| Group | Reviews | negative | neutral | positive |", "|---|---|---|---|---|"]
    for g, mix in cred["sentiment_mix"].items():
        lines.append(f"| {g} | {cred['counts'].get(g, 0):,} | " + " | ".join(f"{mix[k]:.1%}" for k in C.LABELS) + " |")
    if "helpful_votes" in cred:
        lines += ["", "| Group | Mean helpful votes | Share with at least one vote |", "|---|---|---|"]
        for g, hv in cred["helpful_votes"].items():
            lines.append(f"| {g} | {hv['mean']:.2f} | {hv['share_with_votes']:.1%} |")
    if len(examples):
        lines += ["", "### Most-voted contradictory reviews", "",
                  "| Stars | VADER | Helpful votes | Excerpt |", "|---|---|---|---|"]
        for _, e in examples.iterrows():
            excerpt = str(e["excerpt"]).replace("|", "/")
            lines.append(f"| {int(e['rating'])} | {e['vader_compound']:+.2f} | {int(e.get('helpful_vote', 0))} | {excerpt} |")

    lines += ["", "## Top terms", "", "| Rank | Bag-of-Words (count) | TF-IDF (total weight) |", "|---|---|---|"]
    bow, tfidf = top_terms["top_bow"], top_terms["top_tfidf"]
    for i in range(min(len(bow), len(tfidf), 20)):
        lines.append(f"| {i + 1} | {bow.iloc[i]['term']} ({int(bow.iloc[i]['count']):,}) | "
                     f"{tfidf.iloc[i]['term']} ({tfidf.iloc[i]['tfidf_weight']:.1f}) |")
    for k, bg in top_terms.get("top_bigrams", {}).items():
        lines += ["", f"**Top bigrams, {k} reviews:** " + ", ".join(bg["ngram"].tolist())]

    lines += ["", "## Figures", ""]
    for p in sorted(C.FIG_DIR.glob("*.png")):
        lines.append(f"![{p.stem}](figures/{p.name})")
    path = C.REPORTS_DIR / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def update_readme(block: str, readme: Path = C.README_PATH) -> bool:
    """Replace the text between the RESULTS markers. Returns False if the markers are missing."""
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), flags=re.S)
    if not pattern.search(text):
        log.warning("README markers not found; results block not written")
        return False
    readme.write_text(pattern.sub(lambda _: f"{START}\n{block}\n{END}", text), encoding="utf-8")
    return True
