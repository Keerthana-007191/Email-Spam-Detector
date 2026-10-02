"""Feature-importance helpers for linear models and Naive Bayes."""
from __future__ import annotations

import numpy as np


def _weights(clf):
    """Per-feature weight; positive pushes towards spam (class 1)."""
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    if hasattr(clf, "feature_log_prob_"):
        flp = clf.feature_log_prob_
        return flp[1] - flp[0]
    return None


def global_top_features(pipeline, n: int = 20):
    """Terms with the largest weight toward spam and toward legitimate mail."""
    w = _weights(pipeline.named_steps["model"])
    if w is None:
        return None
    names = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    order = np.argsort(w)
    return {
        "spam": [(str(names[i]), float(w[i])) for i in order[::-1][:n]],
        "legitimate": [(str(names[i]), float(w[i])) for i in order[:n]],
    }


def explain_email(pipeline, text: str, n: int = 8):
    """Terms in this email that pushed the score most toward each class
    (TF-IDF value x model weight). Returns None if the model has no weights."""
    w = _weights(pipeline.named_steps["model"])
    if w is None:
        return None
    names = np.asarray(pipeline.named_steps["tfidf"].get_feature_names_out())
    X = pipeline[:-1].transform([text]).tocsr()
    contrib = X.data * w[X.indices]
    order = np.argsort(contrib)
    spam = [(str(names[X.indices[i]]), float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:n]
    ham = [(str(names[X.indices[i]]), float(contrib[i])) for i in order if contrib[i] < 0][:n]
    return {"spam": spam, "legitimate": ham}