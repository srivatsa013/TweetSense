"""
src/model.py
------------
Supervised Learning Module  (SDD §3.4)

Trains SVM (LinearSVC) and Logistic Regression on combined
TF-IDF + engineered feature vectors.

3-Class strategy
----------------
Sentiment140 has only binary labels (Negative=0, Positive=2).
We derive a Neutral class (1) at prediction time using the model's
confidence: if max(probability) < NEUTRAL_THRESHOLD → Neutral.

This is a well-documented approach in Twitter sentiment literature.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_MAP     = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_COLORS  = {"Negative": "#ef4444", "Neutral": "#f59e0b", "Positive": "#22c55e"}
MODEL_DIR     = "models"
NEUTRAL_THRESHOLD = 0.62   # confidence below this → Neutral

# ── TF-IDF config ─────────────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features=30_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    analyzer="word",
)


# ── Vectorization helpers ──────────────────────────────────────────────────────
def build_feature_matrix(
    clean_texts: pd.Series,
    eng_features: pd.DataFrame,
    tfidf: TfidfVectorizer | None,
    fit_tfidf: bool = False,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """
    Combine TF-IDF matrix with engineered feature columns.

    Parameters
    ----------
    clean_texts  : Series of preprocessed tweet strings
    eng_features : DataFrame from features.extract_features()
    tfidf        : existing TfidfVectorizer (None → create new)
    fit_tfidf    : if True, fit the vectorizer on clean_texts

    Returns
    -------
    (sparse_matrix, tfidf_vectorizer)
    """
    if tfidf is None:
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)

    if fit_tfidf:
        tfidf_matrix = tfidf.fit_transform(clean_texts)
    else:
        tfidf_matrix = tfidf.transform(clean_texts)

    # Scale engineered features and convert to sparse
    eng_arr    = eng_features.fillna(0).values.astype(np.float32)
    eng_sparse = csr_matrix(eng_arr)

    return hstack([tfidf_matrix, eng_sparse], format="csr"), tfidf


# ── Training ──────────────────────────────────────────────────────────────────
def train(
    df: pd.DataFrame,
    model_type: str = "svm",
    test_size: float = 0.2,
) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    df         : DataFrame with columns ['text', 'clean_text', 'label',
                 + engineered feature columns]
    model_type : 'svm' | 'lr'
    test_size  : fraction for evaluation split

    Returns
    -------
    dict with keys: model, tfidf, metrics, report, X_test, y_test, y_pred
    """
    from src.features import extract_features
    from src.preprocess import preprocess_dataframe

    eng_features = extract_features(df, text_col="text")

    X_full, tfidf = build_feature_matrix(
        df["clean_text"], eng_features, tfidf=None, fit_tfidf=True
    )
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Model selection ──
    if model_type == "svm":
        base = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        # Wrap with Platt scaling to get probabilities
        clf = CalibratedClassifierCV(base, cv=3)
    else:  # logistic regression
        clf = LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto",
            class_weight="balanced",
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)

    # Raw predictions (binary: 0 or 2)
    y_pred_raw = clf.predict(X_test)

    # Apply neutral zone via confidence
    probas = clf.predict_proba(X_test)
    y_pred = _apply_neutral(probas, y_pred_raw)

    # Metrics (evaluate against binary ground truth because dataset has no neutral)
    # We report accuracy vs. 3-class output for transparency
    acc = accuracy_score(y_test, y_pred_raw)   # binary accuracy
    report = classification_report(
        y_pred,
        y_pred,   # self-report on 3-class distribution for display
        target_names=["Negative", "Neutral", "Positive"],
        labels=[0, 1, 2],
        zero_division=0,
        output_dict=True,
    )
    # Real classification report vs ground truth (binary)
    gt_report = classification_report(
        y_test,
        y_pred_raw,
        target_names=["Negative", "Positive"],
        labels=[0, 2],
        zero_division=0,
    )

    metrics = {
        "accuracy": round(acc * 100, 2),
        "model_type": model_type.upper(),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "classification_report": gt_report,
        "confusion_matrix": confusion_matrix(y_test, y_pred_raw, labels=[0, 2]),
    }

    # ── Persist ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf,   os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.pkl"))

    return {
        "model":    clf,
        "tfidf":    tfidf,
        "metrics":  metrics,
        "X_test":   X_test,
        "y_test":   y_test,
        "y_pred":   y_pred,
        "y_pred_raw": y_pred_raw,
        "eng_features": eng_features,
    }


# ── Prediction ────────────────────────────────────────────────────────────────
def _apply_neutral(probas: np.ndarray, raw_preds: np.ndarray) -> np.ndarray:
    """
    Replace predictions with Neutral (1) where confidence is low.
    probas shape: (n_samples, n_classes) — classes are [0, 2]
    """
    preds = raw_preds.copy()
    max_conf = probas.max(axis=1)
    preds[max_conf < NEUTRAL_THRESHOLD] = 1
    return preds


def predict_single(
    text: str,
    clf,
    tfidf: TfidfVectorizer,
) -> dict:
    """
    Predict sentiment for a single raw tweet string.

    Returns
    -------
    dict: { label_id, label_str, confidence, probas }
    """
    from src.preprocess import clean_tweet
    from src.features import extract_features

    clean = clean_tweet(text)
    tmp_df = pd.DataFrame({"text": [text], "clean_text": [clean]})
    eng    = extract_features(tmp_df, text_col="text")

    X, _ = build_feature_matrix(
        tmp_df["clean_text"], eng, tfidf=tfidf, fit_tfidf=False
    )

    proba    = clf.predict_proba(X)[0]
    raw_pred = clf.predict(X)[0]
    label_id = _apply_neutral(proba[np.newaxis, :], np.array([raw_pred]))[0]
    confidence = float(proba.max())

    return {
        "label_id":   int(label_id),
        "label_str":  LABEL_MAP[int(label_id)],
        "confidence": round(confidence * 100, 1),
        "probas":     {
            "Negative": round(float(proba[0]) * 100, 1),
            "Positive": round(float(proba[-1]) * 100, 1),
        },
    }


def load_model(model_dir: str = MODEL_DIR):
    """Load persisted classifier and TF-IDF vectorizer from disk."""
    clf   = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf.pkl"))
    return clf, tfidf