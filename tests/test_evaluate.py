import pandas as pd

from evaluate import compute_metrics, select_best_model


def test_compute_metrics_known_values():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    m = compute_metrics(y_true, y_pred)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["f1"] == 0.75
    assert m["false_positive_rate"] == 0.25
    assert m["accuracy"] == 0.75


def test_average_precision_present_with_scores():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["average_precision"] == 1.0


def _cv(rows):
    return pd.DataFrame(rows, columns=["model", "f1_mean", "false_positive_rate_mean", "recall_mean"])


def test_tie_breaks_on_lower_false_positive_rate():
    df = _cv([("A", 0.980, 0.010, 0.97), ("B", 0.981, 0.020, 0.98), ("C", 0.950, 0.005, 0.94)])
    assert select_best_model(df) == "A"


def test_clear_winner_is_selected_even_with_higher_fpr():
    df = _cv([("A", 0.980, 0.010, 0.97), ("B", 0.990, 0.020, 0.99)])
    assert select_best_model(df) == "B"