"""Train, compare and persist the spam classifier.

Usage:  python train.py  [--data data/enron_spam_data.csv] [--max-features 50000]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")  # no GUI needed; plots are saved to files

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
import evaluate as ev
from data_loader import LABEL, TEXT, load_dataset
from explain import global_top_features
from preprocess import EmailPreprocessor

SELECTION_RULE = ("Highest mean spam F1 in stratified CV on the training split; models within "
                  "0.002 F1 are tied and the lowest false-positive rate wins, then highest recall.")


def make_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVC": LinearSVC(random_state=config.RANDOM_STATE),
    }


def make_vectorizer(max_features=50000, min_df=2) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                           min_df=min_df, max_df=0.95, sublinear_tf=True)


def build_pipeline(model, max_features=50000, min_df=2) -> Pipeline:
    """Raw email text in -> prediction out. The unit that gets saved."""
    return Pipeline([
        ("preprocess", EmailPreprocessor()),
        ("tfidf", make_vectorizer(max_features, min_df)),
        ("model", model),
    ])


def main(args) -> None:
    t0 = time.time()
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data
    df, report, summary = load_dataset(args.data)
    print("Cleaning report:", asdict(report))
    print("Dataset summary:", summary)

    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT], df[LABEL], test_size=args.test_size,
        stratify=df[LABEL], random_state=config.RANDOM_STATE)
    print(f"\nTrain: {len(X_train)} emails | Test: {len(X_test)} emails "
          f"| spam ratio train {y_train.mean():.4f} / test {y_test.mean():.4f}")

    # 2. Cross-validation on the TRAIN split only.
    # The preprocessor is stateless, so cleaning once up front is leakage-free;
    # TF-IDF and the model are refit inside every fold.
    print("\nCleaning training text and running cross-validation ...")
    X_train_clean = EmailPreprocessor().transform(X_train)
    cv_pipes = {name: Pipeline([("tfidf", make_vectorizer(args.max_features)), ("model", m)])
                for name, m in make_models().items()}
    cv_df = ev.cross_validate_models(cv_pipes, X_train_clean, y_train.to_numpy(),
                                     n_splits=args.cv_folds, n_jobs=args.n_jobs)
    best_name = ev.select_best_model(cv_df)
    show = ["model", "precision_mean", "recall_mean", "f1_mean", "weighted_f1_mean",
            "false_positive_rate_mean"]
    print(f"\nCross-validation ({args.cv_folds}-fold, train split):")
    print(cv_df[show].round(4).to_string(index=False))
    print(f"\nSelected model: {best_name}\nRule: {SELECTION_RULE}")

    # 3. Fit every full pipeline on the train split, report on the held-out test set
    test_rows, cms, curves, pipelines = [], {}, {}, {}
    for name, model in make_models().items():
        pipe = build_pipeline(model, args.max_features).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        scores = ev.get_scores(pipe, X_test)
        m = ev.compute_metrics(y_test, pred, scores)
        test_rows.append({"model": name, **m})
        cms[name] = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        if scores is not None:
            curves[name] = (y_test.to_numpy(), scores)
        pipelines[name] = pipe
    test_df = pd.DataFrame(test_rows)
    cols = ["model", "accuracy", "precision", "recall", "f1", "weighted_f1",
            "false_positive_rate", "average_precision", "fp", "fn"]
    print("\nHeld-out test set:")
    print(test_df[cols].round(4).to_string(index=False))

    # 4. Plots and tables
    out = config.OUTPUT_DIR
    ev.plot_confusion_matrices(cms, out / "confusion_matrices.png")
    ev.plot_pr_curves(curves, out / "pr_curves.png")
    ev.plot_model_comparison(test_df, path=out / "model_comparison.png")
    cv_df.to_csv(out / "cv_results.csv", index=False)
    test_df.to_csv(out / "test_results.csv", index=False)

    # 5. Persist the selected pipeline
    best = pipelines[best_name]
    joblib.dump(best, config.MODEL_PATH, compress=3)
    print(f"\nSaved pipeline -> {config.MODEL_PATH}")

    ev.save_json({
        "selected_model": best_name,
        "selection_rule": SELECTION_RULE,
        "best_by_test_f1": str(test_df.sort_values("f1", ascending=False).iloc[0]["model"]),
        "dataset": {"cleaning_report": asdict(report), "summary": summary,
                    "train_size": int(len(X_train)), "test_size": int(len(X_test))},
        "settings": {"ngram_range": [1, 2], "max_features": args.max_features,
                     "min_df": 2, "max_df": 0.95, "sublinear_tf": True,
                     "cv_folds": args.cv_folds, "test_size": args.test_size,
                     "random_state": config.RANDOM_STATE,
                     "sklearn_version": sklearn.__version__},
        "cv_results": cv_df.to_dict(orient="records"),
        "test_results": test_df.to_dict(orient="records"),
        "confusion_matrices": cms,
        "top_features": global_top_features(best, 20),
    }, out / "metrics.json")
    print(f"Saved metrics and plots -> {out}\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the SpamMail AI classifier")
    p.add_argument("--data", default=None, help="CSV file or folder (default: data/)")
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1, help="CV parallelism (-1 = all cores)")
    main(p.parse_args())