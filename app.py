# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# try to import helpers from core, otherwise provide safe fallbacks
try:
    from core import clean_text, log_prediction, load_artifacts, ARTIFACTS_DIR  # type: ignore
except Exception:
    ARTIFACTS_DIR = "artifacts"

    def clean_text(text: str) -> str:
        """Simple fallback cleaning used by the app if core.clean_text is missing."""
        import re
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def log_prediction(task: str, raw_text: str, label: str, probs) -> None:
        """Fallback logger that appends a CSV entry to artifacts/predictions_log.csv (best-effort)."""
        import time
        import json
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
            # best-effort: don't raise — logging should not break the app
            pass

    def load_artifacts():
        """Attempt to load common artifact files from artifacts/ — return Nones if unavailable."""
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
            # If loading fails, return Nones — app will show friendly errors instead of crashing
            model_sent = model_sent or None
            vectorizer_sent = vectorizer_sent or None
            fake_pipeline = fake_pipeline or None
        return model_sent, vectorizer_sent, fake_pipeline

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Social Analytics — Sentiment & Fake-news", layout="wide")
st.title("🔎 Social Analytics — Sentiment & Fake-news")

# Load artifacts (single call)
model_sent, vectorizer_sent, fake_pipeline = load_artifacts()

# label mapping and names
sent_label_order = sorted(["Negative", "Neutral", "Positive"])
sent_label_mapping = {i: label for i, label in enumerate(sent_label_order)}
fake_label_names = ["Real", "Fake"]  # 0 -> Real, 1 -> Fake

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Options")
    mode = st.selectbox("Product / Task", ["Sentiment analysis", "Fake-news detection"])
    show_metrics = st.checkbox("Show model metrics (quick eval)")
    st.markdown("---")
    st.write("Model availability")
    st.write(f"- Sentiment model: {'✅' if model_sent and vectorizer_sent else '❌'}")
    st.write(f"- Fake-news pipeline: {'✅' if fake_pipeline else '❌'}")
    st.markdown("---")
    display_audit = st.checkbox("Show audit log (last 200)", value=False)

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader(mode)

    # Examples + input box
    if mode == "Sentiment analysis":
        examples = {
            "Positive — happy": "I absolutely love this! Best day ever.",
            "Neutral — factual": "The product arrived yesterday and was packaged well.",
            "Negative — angry": "This is terrible, I'm very disappointed and frustrated."
        }
        choice = st.selectbox("Choose example (or Custom)", ["Custom"] + list(examples.keys()))
        if choice == "Custom":
            user_text = st.text_area("Text input", value="I love this!", height=140)
        else:
            user_text = st.text_area("Text input", value=examples[choice], height=140)

        if st.button("Predict Sentiment"):
            if not user_text.strip():
                st.warning("Type something to predict.")
            elif model_sent is None or vectorizer_sent is None:
                st.error("Sentiment model or vectorizer missing. Place them in artifacts/.")
            else:
                cleaned = clean_text(user_text)
                vec = vectorizer_sent.transform([cleaned])
                pred = model_sent.predict(vec)[0]
                probs = model_sent.predict_proba(vec)[0]

                st.markdown("### Result")
                st.success(f"{sent_label_mapping[pred]}")
                prob_series = pd.Series(data=probs, index=[sent_label_mapping[i] for i in range(len(probs))])
                st.bar_chart(prob_series.sort_values(ascending=False))
                st.write(prob_series.to_frame("confidence"))

                log_prediction("sentiment", user_text, sent_label_mapping[pred], probs)

    else:
        examples_fake = {
            "Fake — sensational": "OOPS: Scientists confirm aliens live among us! (shocking)",
            "Real — newswire": "U.S. President met with foreign leaders to discuss trade deals.",
            "Neutral — factual headline": "Company announces quarterly earnings and revenue results."
        }
        choice2 = st.selectbox("Choose example (or Custom)", ["Custom"] + list(examples_fake.keys()))
        if choice2 == "Custom":
            user_text = st.text_area("Text input (headline/article)", value="Breaking: ...", height=140)
        else:
            user_text = st.text_area("Text input (headline/article)", value=examples_fake[choice2], height=140)

        if st.button("Predict Fake-news"):
            if not user_text.strip():
                st.warning("Type something to predict.")
            elif fake_pipeline is None:
                st.error("Fake-news pipeline not found. Run training notebook and save artifacts/fake_model_skl_pipeline.joblib")
            else:
                cleaned = clean_text(user_text)
                probs = fake_pipeline.predict_proba([cleaned])[0]
                pred = fake_pipeline.predict([cleaned])[0]
                label_name = fake_label_names[int(pred)]

                st.markdown("### Result")
                if pred == 1:
                    st.error(label_name)
                else:
                    st.success(label_name)

                prob_series = pd.Series(data=probs, index=fake_label_names)
                st.bar_chart(prob_series.sort_values(ascending=False))
                st.write(prob_series.to_frame("confidence"))

                log_prediction("fake-news", user_text, label_name, probs)

    # Bulk upload
    st.markdown("---")
    st.subheader("Bulk prediction (CSV upload)")
    uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_upload = None

        if df_upload is not None:
            if "text" not in df_upload.columns:
                st.error("CSV must have a `text` column.")
            else:
                st.info(f"Running predictions on {len(df_upload)} rows...")
                cleaned_list = df_upload['text'].astype(str).apply(clean_text).tolist()

                if mode == "Sentiment analysis":
                    if model_sent is None or vectorizer_sent is None:
                        st.error("Sentiment artifacts missing.")
                    else:
                        vecs = vectorizer_sent.transform(cleaned_list)
                        preds = model_sent.predict(vecs)
                        probs = model_sent.predict_proba(vecs)
                        df_upload['pred'] = [sent_label_mapping[p] for p in preds]
                        for i in range(min(20, len(df_upload))):
                            log_prediction("sentiment-bulk", df_upload.loc[i,'text'], df_upload.loc[i,'pred'], probs[i])
                        st.dataframe(df_upload.head(200))
                        st.markdown("Prediction counts:")
                        st.write(df_upload['pred'].value_counts())
                else:
                    if fake_pipeline is None:
                        st.error("Fake-news pipeline missing.")
                    else:
                        preds = fake_pipeline.predict(cleaned_list)
                        probs = fake_pipeline.predict_proba(cleaned_list)
                        df_upload['pred'] = [fake_label_names[int(p)] for p in preds]
                        for i in range(min(20, len(df_upload))):
                            log_prediction("fake-bulk", df_upload.loc[i,'text'], df_upload.loc[i,'pred'], probs[i])
                        st.dataframe(df_upload.head(200))
                        st.markdown("Prediction counts:")
                        st.write(df_upload['pred'].value_counts())

with col2:
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
                    fig, ax = plt.subplots(figsize=(4,3))
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
            df_log = pd.read_csv(log_path)
            st.dataframe(df_log.tail(200).sort_values("timestamp", ascending=False))
            st.download_button("Download audit CSV", data=open(log_path, "rb"), file_name="predictions_log.csv")
        else:
            st.info("No predictions logged yet. Use the app to generate entries (they are saved to artifacts/predictions_log.csv).")

# Footer
st.markdown("---")
st.caption("Notes: student prototype. For production, add authentication, secure logging, and governance.")
