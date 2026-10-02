"""Email Spam Detector - Streamlit app. Run with:  streamlit run app.py"""
from __future__ import annotations

import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

import config
from explain import explain_email
from preprocess import EmailPreprocessor  # noqa: F401  (needed to unpickle the pipeline)

MAX_CHARS = 100_000
MAX_UPLOAD_BYTES = 1_000_000
METRICS_PATH = config.OUTPUT_DIR / "metrics.json"
PR_IMAGE = config.OUTPUT_DIR / "pr_curves.png"
NAVY, BLUE, RED, GREEN = "#0B2545", "#13315C", "#B42318", "#17603A"

SAMPLES = {
    "Spam: pharmacy offer": (
        "Subject: CHEAP v1agra and c1alis - 90% OFF!!!\n\nOrder now and save $$$ on "
        "prescription meds. No prescription needed! Click http://cheap-pills.example.com "
        "or reply to sales@pills-direct.example.com. Limited time offer!"),
    "Spam: prize notification": (
        "Subject: You have won $1,000,000!\n\nCongratulations! You were selected in our "
        "weekly draw. To claim your prize, send your bank details to "
        "claims@lucky-win.example.com today. Act now, this offer expires soon!!!"),
    "Legitimate: meeting request": (
        "Subject: Schedule for Thursday\n\nHi Sarah, can we move the gas nomination review to "
        "2 pm on Thursday? I have attached the updated schedule. Let me know if that works. "
        "Thanks, Mark"),
    "Legitimate: project update": (
        "Subject: Q3 pipeline report\n\nTeam, please find the Q3 pipeline report attached. "
        "The volumes look in line with forecast and I do not expect any changes before "
        "month end. Happy to discuss on Monday."),
}

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")

st.markdown(f"""
<style>
.block-container {{padding-top: 1.5rem; max-width: 1200px;}}
.hero {{background: linear-gradient(135deg, {NAVY}, {BLUE}); padding: 1.6rem 2rem;
        border-radius: 14px; margin-bottom: 1.2rem;}}
.hero h1 {{color: #fff; margin: 0; font-size: 2rem;}}
.hero p {{color: #C9D6EA; margin: .3rem 0 0 0;}}
.card {{background: #fff; border: 1px solid #D6E0F0; border-left: 5px solid {BLUE};
        border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .6rem;}}
.card .label {{color: #5B6B86; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em;}}
.card .value {{color: {NAVY}; font-size: 1.65rem; font-weight: 700;}}
.result {{border-radius: 12px; padding: 1.1rem 1.4rem; font-size: 1.5rem; font-weight: 700;
          margin-bottom: .8rem;}}
.result.spam {{background: #FDECEC; color: {RED}; border: 1px solid #F5B5B5;}}
.result.ham {{background: #E8F5EC; color: {GREEN}; border: 1px solid #A9D9BB;}}
</style>
<div class="hero"><h1>📧 Email Spam Detector</h1>
<p>NLP and machine-learning spam detection trained on the Enron email corpus</p></div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model ...")
def load_pipeline():
    if not config.MODEL_PATH.exists():
        return None, (f"Model file `{config.MODEL_PATH.name}` was not found in the models "
                      "folder. Run `python train.py` first.")
    try:
        return joblib.load(config.MODEL_PATH), None
    except Exception as exc:  # corrupted file, version mismatch, ...
        return None, f"The model file could not be loaded ({type(exc).__name__}: {exc})."


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


pipe, load_error = load_pipeline()
if pipe is None:
    st.error(load_error)
    st.stop()
metrics = load_metrics()
model_name = metrics["selected_model"] if metrics else type(pipe.named_steps["model"]).__name__


# ------------------------------------------------------------- helpers
def card(label: str, value: str) -> None:
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def analyze(text: str) -> dict:
    pred = int(pipe.predict([text])[0])
    if hasattr(pipe, "decision_function"):
        score = float(pipe.decision_function([text])[0])
        kind, mag = "Decision score", abs(score)
        strength = "Borderline" if mag < 0.5 else "Moderate" if mag < 1.5 else "Strong"
    elif hasattr(pipe, "predict_proba"):
        score = float(pipe.predict_proba([text])[0][1])
        kind, mag = "Spam probability", max(score, 1 - score)
        strength = "Borderline" if mag < 0.7 else "Moderate" if mag < 0.9 else "Strong"
    else:
        score, kind, strength = float("nan"), "Score", "n/a"
    return {"pred": pred, "score": score, "kind": kind, "strength": strength}


def contrib_chart(items, color, title):
    if not items:
        st.caption(f"No terms found for: {title}")
        return
    df = pd.DataFrame(items, columns=["term", "contribution"])
    df["contribution"] = df["contribution"].abs()
    chart = (alt.Chart(df).mark_bar(color=color)
             .encode(x=alt.X("contribution:Q", title="Influence"),
                     y=alt.Y("term:N", sort="-x", title=None),
                     tooltip=["term", alt.Tooltip("contribution:Q", format=".3f")])
             .properties(title=title, height=26 * len(df) + 40))
    st.altair_chart(chart, use_container_width=True)


def read_upload(file) -> str | None:
    if file.size > MAX_UPLOAD_BYTES:
        st.error("That file is larger than 1 MB. Please upload a smaller .txt email.")
        return None
    raw = file.getvalue()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def load_sample() -> None:
    st.session_state["email_text"] = SAMPLES[st.session_state["sample_choice"]]


# ----------------------------------------------------------------- tabs
tab_predict, tab_perf, tab_about = st.tabs(["🔍 Analyze Email", "📊 Model Performance", "ℹ️ About"])

with tab_predict:
    st.session_state.setdefault("email_text", "")
    left, right = st.columns([3, 2], gap="large")

    with right:
        st.subheader("Try a sample")
        st.selectbox("Sample email", list(SAMPLES), key="sample_choice")
        st.button("Load sample", on_click=load_sample)
        st.subheader("Or upload a .txt file")
        up = st.file_uploader("Email file", type=["txt"], label_visibility="collapsed")
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get("upload_sig") != sig:
                content = read_upload(up)
                if content is not None:
                    st.session_state["email_text"] = content
                st.session_state["upload_sig"] = sig
                st.rerun()

    with left:
        st.subheader("Paste an email")
        st.text_area("Email text", key="email_text", height=260,
                     placeholder="Paste the subject and body of an email here ...",
                     label_visibility="collapsed")
        run = st.button("Analyze email", type="primary")

    if run:
        text = st.session_state["email_text"]
        if not text.strip():
            st.warning("Please paste or upload an email first.")
        else:
            if len(text) > MAX_CHARS:
                st.warning(f"Email is very long; only the first {MAX_CHARS:,} characters are used.")
                text = text[:MAX_CHARS]
            res = analyze(text)
            is_spam = res["pred"] == 1
            st.markdown(
                f'<div class="result {"spam" if is_spam else "ham"}">'
                f'{"🚫 Spam" if is_spam else "✅ Legitimate"}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                card(res["kind"], f'{res["score"]:+.2f}' if res["kind"] == "Decision score"
                     else f'{res["score"]:.1%}')
            with c2:
                card("Signal strength", res["strength"])
            with c3:
                card("Model", model_name)
            st.caption(
                "The score is the distance from the model's decision boundary, not a calibrated "
                "probability. Positive means spam, negative means legitimate, and values near 0 "
                "are uncertain. The model learned from 1999-2005 Enron emails, so modern "
                "phishing or unusual writing styles may be misjudged."
                if res["kind"] == "Decision score" else
                "Probabilities from Naive Bayes are often over-confident; treat them as a rough guide.")

            exp = explain_email(pipe, text)
            if exp is None:
                st.info("This model type does not expose per-term influence.")
            else:
                st.markdown("#### Why this prediction?")
                e1, e2 = st.columns(2)
                with e1:
                    contrib_chart(exp["spam"], RED, "Terms pushing toward spam")
                with e2:
                    contrib_chart(exp["legitimate"], GREEN, "Terms pushing toward legitimate")
                st.caption("Terms are shown after preprocessing: urltoken, emailtoken, numtoken, "
                           "dollartoken and exclaimtoken stand for URLs, email addresses, numbers, "
                           "$ signs and ! marks.")

with tab_perf:
    if not metrics:
        st.info("No `outputs/metrics.json` found. Run `python train.py` to generate it.")
    else:
        test_df = pd.DataFrame(metrics["test_results"])
        cv_df = pd.DataFrame(metrics["cv_results"])
        sel = test_df[test_df["model"] == metrics["selected_model"]].iloc[0]

        st.markdown(f"**Selected model: {metrics['selected_model']}** "
                    f"(held-out test set: {metrics['dataset']['test_size']:,} emails)")
        cols = st.columns(5)
        for col, (lab, val) in zip(cols, [
                ("Accuracy", f"{sel['accuracy']:.2%}"), ("Spam precision", f"{sel['precision']:.2%}"),
                ("Spam recall", f"{sel['recall']:.2%}"), ("Weighted F1", f"{sel['weighted_f1']:.4f}"),
                ("False-positive rate", f"{sel['false_positive_rate']:.2%}")]):
            with col:
                card(lab, val)
        st.caption(f"Selection rule: {metrics['selection_rule']}")

        st.markdown("#### Model comparison (test set)")
        long = test_df.melt(id_vars="model", value_vars=["precision", "recall", "f1", "weighted_f1"],
                            var_name="metric", value_name="score")
        comp = (alt.Chart(long).mark_bar()
                .encode(x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
                        xOffset="metric:N",
                        y=alt.Y("score:Q", scale=alt.Scale(domain=[0.97, 1.0]), title="Score"),
                        color=alt.Color("metric:N", scale=alt.Scale(
                            range=["#9DB7E0", "#13315C", "#4C78C5", "#0B2545"])),
                        tooltip=["model", "metric", alt.Tooltip("score:Q", format=".4f")])
                .properties(height=320).interactive())
        st.altair_chart(comp, use_container_width=True)
        st.caption("The y-axis starts at 0.97 to make small differences visible; "
                   "all models score above this.")

        st.markdown("#### Confusion matrix")
        which = st.selectbox("Model", list(metrics["confusion_matrices"]),
                             index=list(metrics["confusion_matrices"]).index(metrics["selected_model"]))
        (tn, fp), (fn, tp) = metrics["confusion_matrices"][which]
        cm = pd.DataFrame({"Actual": ["Legitimate", "Legitimate", "Spam", "Spam"],
                           "Predicted": ["Legitimate", "Spam", "Legitimate", "Spam"],
                           "Count": [tn, fp, fn, tp]})
        base = alt.Chart(cm).encode(x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("Actual:N"))
        heat = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"),
                                                       legend=None))
        txt = base.mark_text(fontSize=22, fontWeight="bold").encode(
            text="Count:Q", color=alt.condition(alt.datum.Count > cm["Count"].max() / 2,
                                                alt.value("white"), alt.value("black")))
        st.altair_chart((heat + txt).properties(width=380, height=300))
        st.caption(f"{fp} legitimate emails were flagged as spam (false positives) and "
                   f"{fn} spam emails were missed (false negatives).")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Cross-validation (5-fold, train split)")
            show = cv_df[["model", "precision_mean", "recall_mean", "f1_mean",
                          "false_positive_rate_mean"]].round(4)
            show.columns = ["Model", "Precision", "Recall", "F1", "FPR"]
            st.dataframe(show, hide_index=True, use_container_width=True)
        with right:
            st.markdown("#### Precision-recall curves")
            if PR_IMAGE.exists():
                st.image(str(PR_IMAGE), use_container_width=True)
            else:
                st.caption("pr_curves.png not found in outputs/.")

        tf = metrics.get("top_features")
        if tf:
            st.markdown("#### Most influential features (selected model)")
            f1c, f2c = st.columns(2)
            with f1c:
                contrib_chart(tf["spam"][:15], RED, "Strongest spam indicators")
            with f2c:
                contrib_chart(tf["legitimate"][:15], GREEN, "Strongest legitimate indicators")

with tab_about:
    st.markdown(f"""
**Email Spam Detector** classifies an email as spam or legitimate.

**Pipeline:** custom email-aware preprocessing (HTML removal, URL and email tokens, negation-safe
stopword removal) → TF-IDF with unigrams and bigrams → {model_name}. The same saved scikit-learn
pipeline is used for training and prediction, so there is no mismatch between the two.

**Data:** the Enron spam corpus (about 30k emails after removing duplicates).

**Limitations**
- Trained on 1999-2005 corporate email; modern spam and phishing may look different.
- Company-specific words in the data can act as shortcuts the model learns.
- English only, and plain text only (no attachments or headers).
- The score is not a calibrated probability.
""")