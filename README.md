---
title: Email Spam Detector
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# 📧 Email Spam Detector

An end-to-end NLP and machine-learning project that classifies emails as **spam** or **legitimate**, with a Streamlit web app.

**Live demo:** _add your Hugging Face Space link here_

## Problem statement
Spam wastes time and carries phishing risk, but a filter that flags real emails is worse than none. This project builds a classifier that catches almost all spam while keeping false positives (legitimate emails marked as spam) low.

## Objectives
- Build a reusable, email-aware preprocessing module.
- Compare Logistic Regression, Multinomial Naive Bayes and Linear SVC fairly.
- Select the final model from measured results, with spam recall and false positives in focus.
- Ship one saved pipeline that is identical in training and inference, and deploy it.

## Dataset
- **Source:** Enron-Spam corpus (Metsis et al., 2006), loaded from the Hugging Face dataset `SetFit/enron_spam` (train and test splits merged).
- **Label mapping:** ham = 0, spam = 1 (spam is the positive class).
- **Cleaning:** 33,716 emails loaded → 51 empty removed → 3,203 exact duplicates removed → **30,462 emails** (15,910 ham, 14,552 spam, 47.8% spam). No conflicting labels were found.
- Subject and body are joined into one text field.
- The dataset is not included in this repo. Run `python get_data.py` to download it.

## Architecture
```
raw email text
  → EmailPreprocessor   (custom, stateless)
  → TfidfVectorizer     (1-2 grams, max_features=50,000, min_df=2, max_df=0.95, sublinear_tf)
  → classifier          (selected by cross-validation)
  → saved as one joblib Pipeline → loaded once by the Streamlit app
```
Because the preprocessor, vectorizer and model sit in one scikit-learn `Pipeline`, training and inference use identical code, and the vectorizer is fitted only on training data (no leakage).

## Technologies
Python, Pandas, NumPy, scikit-learn, NLTK, Matplotlib, Seaborn, Altair, Streamlit, joblib, pytest, Hugging Face Spaces.

## NLP preprocessing
Lowercasing → HTML, script and style removal (URLs inside tags are kept) → URLs replaced by `urltoken`, email addresses by `emailtoken` → `$`, `!` and numbers kept as `dollartoken`, `exclaimtoken` and `numtoken` → contractions expanded (`can't` → `cannot`) → punctuation removed → stopwords removed **except negations** (`not`, `no`, `never`, ...). Empty, `None`, NaN, bytes and malformed input are handled, and very long emails are truncated at 20,000 characters.

## Model comparison methodology
- 80/20 stratified train/test split, `random_state=42` (24,369 train / 6,093 test).
- **Selection used only the training split:** 5-fold stratified cross-validation. Rule: highest mean spam F1; models within 0.002 F1 are tied and the lowest false-positive rate wins.
- The test set was used once, for the final report.

## Results

**Cross-validation (5-fold, train split)**

| Model | Precision | Recall | F1 | Weighted F1 | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 0.9763 | 0.9939 | 0.9850 | 0.9856 | 0.0221 |
| Multinomial Naive Bayes | 0.9874 | 0.9859 | 0.9867 | 0.9873 | 0.0115 |
| **Linear SVC** | 0.9872 | 0.9942 | 0.9907 | 0.9911 | 0.0118 |

**Held-out test set (6,093 emails)**

| Model | Accuracy | Precision | Recall | F1 | Weighted F1 | FPR | Avg. precision | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9892 | 0.9834 | 0.9942 | 0.9887 | 0.9892 | 0.0154 | 0.9983 | 49 | 17 |
| Multinomial Naive Bayes | 0.9865 | 0.9889 | 0.9828 | 0.9859 | 0.9865 | 0.0101 | 0.9984 | 32 | 50 |
| **Linear SVC** | **0.9933** | 0.9908 | **0.9952** | **0.9930** | **0.9933** | **0.0085** | **0.9993** | 27 | 14 |

**Selected model: Linear SVC**, with the best CV F1, the best test F1 and recall, and the lowest false-positive rate on the test set. Naive Bayes was competitive on precision but missed 50 spam emails versus 14.

## Screenshots
_Add images to the `screenshots/` folder, then link them here._

## Installation and usage
```bash
git clone https://github.com/Keerthana-007191/Email-Spam-Detector.git
cd spam-mail-detection
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python get_data.py            # downloads the dataset to data/
python train.py               # trains, evaluates, saves models/ and outputs/
pytest -v                     # 35 tests
streamlit run app.py
```

## Deployment (Hugging Face Spaces)
The Space uses the Streamlit SDK with `app.py` as the entry point. The trained pipeline (`models/spam_pipeline.joblib`) and `outputs/` are committed, so the app starts without retraining.

## Limitations and future work
- Trained on 1999-2005 corporate email; modern spam and phishing may differ.
- Enron-specific vocabulary can act as shortcuts, so real-world accuracy will likely be lower than the figures above.
- English text only; no headers, attachments or images.
- The score is a decision distance, not a calibrated probability.
- Future work: probability calibration, a more recent multi-source dataset, transformer embeddings, header features, and drift monitoring.