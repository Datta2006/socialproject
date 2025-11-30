# app.py
"""
Combined Streamlit app: Sentiment + Fake-news in a single pane with presets, batch CSV,
download, quick metrics and an audit log. Drop your model artifacts into `artifacts/`:
 - sentiment_model.joblib
 - vectorizer.joblib
 - fake_model_skl_pipeline.joblib

This file is a self-contained replacement for your previous app.py (keeps same helper fallbacks).
Copy & paste into your project root.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import io
import time
import json

# -------------------------
# Try to import core helpers; fall back to safe implementations if missing
# -------------------------
try:
    from core import clean_text, log_prediction, load_artifacts, ARTIFACTS_DIR  # type: ignore
except Exception:
    ARTIFACTS_DIR = "artifacts"

    def clean_text(text: str) -> str:
        import re
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def log_prediction(task: str, raw_text: str, label: str, probs) -> None:
        try:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            log_path = os.path.join(ARTIFACTS_DIR, "predictions_log.csv")
            row = {
                "timestamp": int(time.time()),
                "task": task,
                "text": raw_text,
                "label": label,
                "probs": json.dumps(probs.tolist() if hasattr(probs, "tolist") else list(probs)),
            }
            header = not os.path.exists(log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                if header:
                    f.write(",".join(row.keys()) + "\n")
                f.write(",".join('"' + str(v).replace('"', '""') + '"' for v in row.values()) + "\n")
        except Exception:
            pass

    def load_artifacts():
        model_sent = None
        vectorizer_sent = None
        fake_pipeline = None
        try:
            sent_model_path = os.path.join(ARTIFACTS_DIR, "sentiment_model.joblib")
            vec_path = os.path.join(ARTIFACTS_DIR, "vectorizer.joblib")
            fake_path = os.path.join(ARTIFACTS_DIR, "fake_model_skl_pipeline.joblib")

            if os.path.exists(sent_model_path):
                model_sent = joblib.load(sent_model_path)
            if os.path.exists(vec_path):
                vectorizer_sent = joblib.load(vec_path)
            if os.path.exists(fake_path):
                fake_pipeline = joblib.load(fake_path)
        except Exception:
            model_sent = model_sent or None
            vectorizer_sent = vectorizer_sent or None
            fake_pipeline = fake_pipeline or None
        return model_sent, vectorizer_sent, fake_pipeline

# -------------------------
# Page config / styling
# -------------------------
st.set_page_config(page_title="Social Analytics — Sentiment & Fake-news", layout="wide")
st.markdown(
    """
    <style>
      .big-title {font-size:28px; font-weight:700;}
      .muted {color: #6b6b6b; font-size:12px}
      .card {background: linear-gradient(180deg,#ffffff,#f7f7ff); padding:10px; border-radius:10px}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='big-title'>🔎 Social Analytics — Sentiment & Fake-news</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>One-pane workflow: pick a preset, paste text, or upload a CSV. Predict both sentiment & fake-news together.</div>", unsafe_allow_html=True)

# -------------------------
# Load models once
# -------------------------
model_sent, vectorizer_sent, fake_pipeline = load_artifacts()

sent_label_order = sorted(["Negative", "Neutral", "Positive"])
sent_label_mapping = {i: label for i, label in enumerate(sent_label_order)}
fake_label_names = ["Real", "Fake"]  # 0 -> Real, 1 -> Fake

# -------------------------
# Preset headlines / samples (includes many examples you provided)
# -------------------------
SAMPLE_HEADLINES = {
    "Trump NYE Embarrassing Message (op-ed style)": (
        "Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. "
        "Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media. "
        "2018 will be a great year for America!"
    ),
    "Papadopoulos Bar Tip Starts Probe (investigation)": (
        "Former Trump campaign adviser George Papadopoulos was drunk in a wine bar when he revealed knowledge "
        "of Russian opposition research on Hillary Clinton, prompting parts of the Russia probe."
    ),
    "Roy Moore Rally Billboard (protest)": (
        "Liberal group used Ivanka Trump's quote on Roy Moore displayed on a mobile billboard outside the rally."
    ),
    "Don Jr Mocked Franken (twitter backlash)": (
        "Donald Trump Jr. mocked Al Franken's resignation on Twitter and was immediately backfired by users."
    ),
    "Walter Scott Sentencing (police brutality)": (
        "Cop Finally Gets His Due, Walter Scott’s Killer Sentenced To Prison — part of ongoing police brutality conversation."
    ),
    "Trump Recognizes Jerusalem (foreign policy controversy)": (
        "President Donald Trump plans to formally recognize Jerusalem as the capital of Israel, a move with global reaction."
    ),
    "Coal CEO Upset (economy & promises)": (
        "Trump supporting coal CEO says tax changes will wipe out thousands of coal mining jobs despite campaign promises."
    ),
    "Jeff Flake Donates to Democrat (GOP dissent)": (
        "Sitting GOP Senator Jeff Flake donates to Alabama Democrat - a sign GOP is split on Roy Moore support."
    ),
    "Transgender Recruits Allowed (Pentagon)": (
        "Transgender people will be allowed to enlist in the U.S. military starting Monday as ordered by federal courts."
    ),
    "Mueller Investigation Should Continue (Senator Graham)": (
        "Lindsey Graham says Special Counsel Mueller should be allowed to do his job without political interference."
    ),
    "US Post Office & Amazon (tweet dispute)": (
        "Trump calls on U.S. Postal Service to charge much more for Amazon shipments, stirring debate on postal finances."
    ),
    "Budget Fight Looms (Reuters politics)": (
        "Budget fight looms as Republicans flip their fiscal script; lawmakers brace to pass a federal budget in January."
    ),
    "Generic Fake Sensational (toy sample)": (
        "Miracle cure found — doctors stunned by instant recovery!"
    ),
}

# -------------------------
# Sidebar: options
# -------------------------
with st.sidebar:
    st.header("Options")
    show_metrics = st.checkbox("Show model metrics (quick eval)")
    display_audit = st.checkbox("Show audit log (last 200)", value=False)
    st.markdown("---")
    st.write("Model availability")
    st.write(f"- Sentiment model: {'ok' if model_sent and vectorizer_sent else 'invalid'}")
    st.write(f"- Fake-news pipeline: {'ok' if fake_pipeline else 'invalid'}")
    st.markdown("---")
    st.caption("Tip: place artifacts in `artifacts/` and a sample metrics CSV at `data/fake.csv` to enable quick eval.")

# -------------------------
# Main layout (single pane with input on left, diagnostics on right)
# -------------------------
col_main, col_diag = st.columns([2, 1], gap="large")

with col_main:
    st.subheader("Input & Predictions")
    input_mode = st.radio("Input mode", ["Single text", "Pick presets", "Upload CSV (batch)"])

    if input_mode == "Single text":
        with st.expander("Quick presets"):
            pcols = st.columns(3)
            keys = list(SAMPLE_HEADLINES.keys())
            for i, k in enumerate(keys):
                if pcols[i % 3].button(k):
                    st.session_state['user_text'] = SAMPLE_HEADLINES[k]

        user_text = st.text_area("Enter headline / text", value=st.session_state.get('user_text', ''), height=180)
        if st.button("Predict Sentiment + Fake-news"):
            if not user_text.strip():
                st.warning("Type or paste some text to predict.")
            else:
                cleaned = clean_text(user_text)

                # Sentiment
                if model_sent and vectorizer_sent:
                    try:
                        vec = vectorizer_sent.transform([cleaned])
                        s_pred = int(model_sent.predict(vec)[0])
                        s_probs = model_sent.predict_proba(vec)[0]
                        s_label = sent_label_mapping[s_pred]
                    except Exception as e:
                        s_label = None
                        s_probs = None
                        st.error(f"Sentiment prediction failed: {e}")
                else:
                    s_label = None
                    s_probs = None

                # Fake-news
                if fake_pipeline:
                    try:
                        f_pred = int(fake_pipeline.predict([cleaned])[0])
                        f_probs = fake_pipeline.predict_proba([cleaned])[0]
                        f_label = fake_label_names[f_pred]
                    except Exception as e:
                        f_label = None
                        f_probs = None
                        st.error(f"Fake-news prediction failed: {e}")
                else:
                    f_label = None
                    f_probs = None

                # Logging (best-effort)
                try:
                    if s_label is not None:
                        log_prediction("sentiment", user_text, s_label, s_probs or [])
                    if f_label is not None:
                        log_prediction("fake-news", user_text, f_label, f_probs or [])
                except Exception:
                    pass

                # Show combined results
                st.markdown("---")
                st.subheader("Combined result")
                r0, r1, r2 = st.columns([1, 1, 1])
                if s_label:
                    emoji = "🙂" if s_label == "Positive" else ("😐" if s_label == "Neutral" else "😞")
                    r0.metric("Sentiment", f"{s_label} {emoji}")
                    pser = pd.Series(data=s_probs, index=[sent_label_mapping[i] for i in range(len(s_probs))])
                    r0.bar_chart(pser.sort_values(ascending=False))
                else:
                    r0.info("Sentiment model not available")

                if f_label:
                    r1.metric("Fake-news", f"{f_label}")
                    pser_f = pd.Series(data=f_probs, index=fake_label_names)
                    r1.bar_chart(pser_f.sort_values(ascending=False))
                else:
                    r1.info("Fake-news model not available")

                r2.markdown("**Input**")
                r2.write(user_text)

    elif input_mode == "Pick presets":
        st.subheader("Select sample headlines")
        selected = st.multiselect("Choose samples to run", options=list(SAMPLE_HEADLINES.keys()), default=list(SAMPLE_HEADLINES.keys())[:3])
        if st.button("Run on selected samples"):
            rows = []
            for key in selected:
                txt = SAMPLE_HEADLINES[key]
                cleaned = clean_text(txt)
                s_label = None; f_label = None; s_probs = None; f_probs = None
                if model_sent and vectorizer_sent:
                    try:
                        vec = vectorizer_sent.transform([cleaned])
                        s_pred = int(model_sent.predict(vec)[0])
                        s_probs = model_sent.predict_proba(vec)[0]
                        s_label = sent_label_mapping[s_pred]
                    except Exception:
                        s_label = None
                if fake_pipeline:
                    try:
                        f_pred = int(fake_pipeline.predict([cleaned])[0])
                        f_probs = fake_pipeline.predict_proba([cleaned])[0]
                        f_label = fake_label_names[f_pred]
                    except Exception:
                        f_label = None
                rows.append({"sample": key, "text": txt, "sentiment": s_label, "fake": f_label})
                try:
                    if s_label is not None:
                        log_prediction("sentiment-sample", txt, s_label, s_probs or [])
                    if f_label is not None:
                        log_prediction("fake-sample", txt, f_label, f_probs or [])
                except Exception:
                    pass

            df = pd.DataFrame(rows)
            st.dataframe(df)
            if 'sentiment' in df.columns and df['sentiment'].notna().any():
                st.markdown("**Sentiment distribution**")
                st.write(df['sentiment'].value_counts(dropna=True))
            if 'fake' in df.columns and df['fake'].notna().any():
                st.markdown("**Fake vs Real**")
                st.write(df['fake'].value_counts(dropna=True))

    else:  # CSV upload
        st.subheader("Bulk prediction: upload CSV")
        uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_upload = None

            if df_upload is not None:
                if "text" not in df_upload.columns:
                    st.error("CSV must contain a `text` column.")
                else:
                    st.info(f"Running predictions on {len(df_upload)} rows...")
                    cleaned_list = df_upload['text'].astype(str).apply(clean_text).tolist()
                    # Sentiment
                    if model_sent and vectorizer_sent:
                        try:
                            vecs = vectorizer_sent.transform(cleaned_list)
                            preds = model_sent.predict(vecs)
                            probs = model_sent.predict_proba(vecs)
                            df_upload['sentiment'] = [sent_label_mapping[int(p)] for p in preds]
                        except Exception as e:
                            st.error(f"Sentiment batch prediction failed: {e}")
                            df_upload['sentiment'] = None
                    else:
                        df_upload['sentiment'] = None
                    # Fake-news
                    if fake_pipeline:
                        try:
                            preds_f = fake_pipeline.predict(cleaned_list)
                            probs_f = fake_pipeline.predict_proba(cleaned_list)
                            df_upload['fake'] = [fake_label_names[int(p)] for p in preds_f]
                        except Exception as e:
                            st.error(f"Fake-news batch prediction failed: {e}")
                            df_upload['fake'] = None
                    else:
                        df_upload['fake'] = None

                    # Log up to 20 rows
                    for i in range(min(20, len(df_upload))):
                        try:
                            log_prediction("bulk", df_upload.loc[i, 'text'], str(df_upload.loc[i, 'sentiment']), [])
                        except Exception:
                            pass

                    st.dataframe(df_upload.head(200))
                    st.markdown("**Prediction counts**")
                    if 'sentiment' in df_upload.columns:
                        st.write(df_upload['sentiment'].value_counts(dropna=True))
                    if 'fake' in df_upload.columns:
                        st.write(df_upload['fake'].value_counts(dropna=True))

                    # Download CSV
                    to_download = io.StringIO()
                    df_upload.to_csv(to_download, index=False)
                    st.download_button("Download predictions CSV", data=to_download.getvalue(), file_name="predictions.csv")

with col_diag:
    st.subheader("Models & Diagnostics")
    st.markdown("**Loaded models**")
    st.write(f"- Sentiment model: {'Loaded' if model_sent and vectorizer_sent else 'Missing'}")
    st.write(f"- Fake-news pipeline: {'Loaded' if fake_pipeline else 'Missing'}")

    st.markdown("---")
    if show_metrics:
        st.subheader("Quick metrics (sample)")
        metrics_path = os.path.join("data", "fake.csv")
        if os.path.exists(metrics_path):
            try:
                df_metrics = pd.read_csv(metrics_path)
                st.info("Computing sample evaluation on data/fake.csv (small subset)...")
                n = min(2000, len(df_metrics))
                sample = df_metrics.sample(n=n, random_state=42)
                txt_col = 'text' if 'text' in sample.columns else ('title' if 'title' in sample.columns else sample.columns[0])
                label_col = 'label' if 'label' in sample.columns else sample.columns[-1]
                sample_texts = sample[txt_col].astype(str).apply(clean_text).tolist()
                sample_labels = sample[label_col].tolist()
                if fake_pipeline:
                    preds = fake_pipeline.predict(sample_texts)
                    st.text(classification_report(sample_labels, preds))
                    cm = confusion_matrix(sample_labels, preds)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
                else:
                    st.info("Fake pipeline not loaded — cannot compute metrics.")
            except Exception as e:
                st.error(f"Could not compute metrics: {e}")
        else:
            st.info("Put a CSV at data/fake.csv to enable quick metrics.")

    st.markdown("---")
    st.subheader("Audit log")
    if display_audit:
        log_path = os.path.join(ARTIFACTS_DIR, "predictions_log.csv")
        if os.path.exists(log_path):
            try:
                df_log = pd.read_csv(log_path)
                # show most recent 200
                st.dataframe(df_log.tail(200).sort_values("timestamp", ascending=False))
                st.download_button("Download audit CSV", data=open(log_path, "rb"), file_name="predictions_log.csv")
            except Exception as e:
                st.error(f"Could not read log: {e}")
        else:
            st.info("No predictions logged yet. Use the app to generate entries (they are saved to artifacts/predictions_log.csv).")

# Footer
st.markdown("---")
st.caption("Student prototype — for production, add authentication, secure logging, and governance.")
