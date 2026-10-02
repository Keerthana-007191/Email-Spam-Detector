import joblib
import pytest

import config
from train import build_pipeline, make_models

TRAIN = ["free money now", "meeting at noon", "win free prize", "project report attached"] * 3
LABELS = [1, 0, 1, 0] * 3

needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                 reason="Run train.py first to create the model")


def test_vocabulary_comes_only_from_training_data():
    pipe = build_pipeline(make_models()["Linear SVC"]).fit(TRAIN, LABELS)
    pipe.predict(["zebra zebra zebra"])  # unseen word at inference time
    assert "zebra" not in pipe.named_steps["tfidf"].vocabulary_


def test_pipeline_accepts_unseen_emails():
    pipe = build_pipeline(make_models()["Logistic Regression"]).fit(TRAIN, LABELS)
    pred = pipe.predict(["totally new email text", "another unseen message"])
    assert len(pred) == 2 and set(pred) <= {0, 1}


def test_pipeline_survives_malformed_input():
    pipe = build_pipeline(make_models()["Multinomial Naive Bayes"]).fit(TRAIN, LABELS)
    pred = pipe.predict(["", None, "   ", "<<<>>>", "http://", "\x00\x01"])
    assert len(pred) == 6


@needs_model
def test_saved_model_predicts_on_unseen_emails():
    model = joblib.load(config.MODEL_PATH)
    pred = model.predict(["Hello, please see the attached report.", "", "<b>hi</b>"])
    assert len(pred) == 3 and set(pred) <= {0, 1}


@needs_model
def test_saved_model_flags_obvious_spam_and_ham():
    model = joblib.load(config.MODEL_PATH)
    spam = "CHEAP v1agra and c1alis!!! Order now $0.99 per pill http://pills.example.com"
    ham = "Please find attached the gas nomination schedule for tomorrow. Thanks, Vince"
    assert model.predict([spam])[0] == 1
    assert model.predict([ham])[0] == 0