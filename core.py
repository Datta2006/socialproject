# core.py
"""
Core utilities for Social Analytics app.
- loading model artifacts
- text cleaning
- audit logging
- small helpers
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Any

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
PREDICTIONS_LOG = os.path.join(ARTIFACTS_DIR, "predictions_log.csv")


def clean_text(text: str) -> str:
    """Shared lightweight text cleaning used before vectorization/prediction."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    # keep alphanumerics and spaces (news often has numbers)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def log_prediction(product: str, input_text: str, pred_label: str, probs: Any) -> None:
    """
    Append a prediction record to artifacts/predictions_log.csv
    probs typically a numpy array or list; stored as JSON-ish string.
    """
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "product": product,
        "input": input_text,
        "prediction": pred_label,
        "probs": (probs.tolist() if hasattr(probs, "tolist") else str(probs))
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(PREDICTIONS_LOG):
        df_row.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PREDICTIONS_LOG, index=False)


def load_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Attempt to load three artifacts:
      - sentiment model (joblib) and its vectorizer
      - fake-news pipeline (joblib)
    Returns (sent_model, sent_vectorizer, fake_pipeline)
    """
    sent_model = None
    sent_vect = None
    fake_pipe = None

    # sentiment artifacts
    try:
        sent_model = joblib.load(os.path.join(ARTIFACTS_DIR, "sentiment_model.joblib"))
        sent_vect = joblib.load(os.path.join(ARTIFACTS_DIR, "vectorizer.joblib"))
    except Exception:
        sent_model = None
        sent_vect = None

    # fake-news: prefer sklearn pipeline artifact name we used earlier
    try:
        fake_pipe = joblib.load(os.path.join(ARTIFACTS_DIR, "fake_model_skl_pipeline.joblib"))
    except Exception:
        # try other names (backwards compatibility)
        for alt in ["fake_model.joblib", "fake_model_rf.joblib", "fake_model_skl_pipeline.joblib"]:
            try:
                fake_pipe = joblib.load(os.path.join(ARTIFACTS_DIR, alt))
                if fake_pipe is not None:
                    break
            except Exception:
                fake_pipe = None

    return sent_model, sent_vect, fake_pipe


def save_joblib(obj: Any, name: str) -> str:
    """Save an object using joblib in artifacts and return path."""
    path = os.path.join(ARTIFACTS_DIR, name)
    joblib.dump(obj, path)
    return path


def load_csv_preview(path: str, n: int = 5) -> pd.DataFrame:
    """Small helper to safely read CSV for preview (used in metrics panel)."""
    df = pd.read_csv(path, nrows=n)
    return df
