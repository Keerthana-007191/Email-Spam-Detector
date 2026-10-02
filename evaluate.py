"""Metrics, cross-validation, model selection and plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

import config

CV_METRICS = ["accuracy", "precision", "recall", "f1", "weighted_f1", "false_positive_rate"]


# ---------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """Spam (label 1) is the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    return out


def get_scores(model, X) -> Optional[np.ndarray]:
    """Continuous spam score: decision_function if available, else P(spam)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return None


def evaluate_model(model, X, y) -> dict:
    """Fit-free evaluation of an already-trained model on (X, y)."""
    return compute_metrics(y, model.predict(X), get_scores(model, X))


# ------------------------------------------------------- cross-validation
def _fpr(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def cross_validate_models(pipelines: Dict[str, object], X, y, n_splits: int = 5,
                          random_state: int = config.RANDOM_STATE,
                          n_jobs: int = 1) -> pd.DataFrame:
    """Stratified k-fold CV. Each fold refits the whole pipeline on its own training part."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "weighted_f1": "f1_weighted",
        "false_positive_rate": make_scorer(_fpr),
    }
    rows = []
    for name, pipe in pipelines.items():
        res = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs)
        row = {"model": name}
        for m in CV_METRICS:
            vals = res[f"test_{m}"]
            row[f"{m}_mean"], row[f"{m}_std"] = float(vals.mean()), float(vals.std())
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_model(cv_df: pd.DataFrame, tolerance: float = 0.002) -> str:
    """Highest mean spam F1; models within `tolerance` are tied and the lowest
    false-positive rate wins (then the highest spam recall)."""
    best_f1 = cv_df["f1_mean"].max()
    tied = cv_df[cv_df["f1_mean"] >= best_f1 - tolerance]
    tied = tied.sort_values(["false_positive_rate_mean", "recall_mean"],
                            ascending=[True, False])
    return str(tied.iloc[0]["model"])


# ------------------------------------------------------------------ plots
def _finish(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(cms: Dict[str, np.ndarray], path=None):
    """cms: model name -> 2x2 array [[tn, fp], [fn, tp]]."""
    labels = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]
    fig, axes = plt.subplots(1, len(cms), figsize=(4.5 * len(cms), 4), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    return _finish(fig, path)


def plot_pr_curves(curves: Dict[str, tuple], path=None):
    """curves: model name -> (y_true, spam_scores)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (y_true, scores) in curves.items():
        p, r, _ = precision_recall_curve(y_true, scores)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, scores):.4f})")
    ax.set_xlabel("Recall (spam)")
    ax.set_ylabel("Precision (spam)")
    ax.set_title("Precision-Recall curves")
    ax.set_ylim(0.8, 1.01)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def plot_model_comparison(df: pd.DataFrame, metrics=("precision", "recall", "f1", "weighted_f1"),
                          path=None):
    """df: one row per model with a 'model' column and one column per metric."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.set_index("model")[list(metrics)].plot.bar(ax=ax, rot=0)
    lo = float(df[list(metrics)].min().min())
    ax.set_ylim(max(0.0, lo - 0.02), 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (held-out test set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, path)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")