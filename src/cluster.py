"""
src/cluster.py
--------------
Clustering Module  (SDD §3.5) + Analysis Module  (SDD §3.6)

Applies K-Means to TF-IDF vectors and enriches clusters with:
  - Cluster size
  - Top keywords per cluster
  - Sentiment distribution per cluster
  - Dominant sentiment label
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from src.model import LABEL_MAP, LABEL_COLORS


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_K   = 5
RANDOM_SEED = 42
SVD_DIMS    = 100   # LSA dimensionality before clustering (speeds up K-Means)


# ── Clustering ────────────────────────────────────────────────────────────────
def cluster_tweets(
    tfidf_matrix,             # sparse (n_samples, vocab)
    k: int = DEFAULT_K,
    use_minibatch: bool = True,
) -> tuple[np.ndarray, object]:
    """
    Run K-Means on LSA-reduced TF-IDF vectors.

    Parameters
    ----------
    tfidf_matrix : sparse TF-IDF feature matrix
    k            : number of clusters
    use_minibatch: use MiniBatchKMeans for speed on large datasets

    Returns
    -------
    (labels, kmeans_model)
        labels: array of cluster ids  (0 … k-1), shape (n_samples,)
    """
    # Reduce dimensionality with LSA (Latent Semantic Analysis)
    svd    = TruncatedSVD(n_components=SVD_DIMS, random_state=RANDOM_SEED)
    norm   = Normalizer(copy=False)
    lsa    = make_pipeline(svd, norm)

    X_lsa  = lsa.fit_transform(tfidf_matrix)

    if use_minibatch:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            batch_size=2048,
            n_init=3,
            max_iter=100,
        )
    else:
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10,
            max_iter=300,
        )

    labels = km.fit_predict(X_lsa)
    return labels, km, lsa, svd


# ── Top keywords per cluster ──────────────────────────────────────────────────
def top_keywords(
    tfidf_matrix,
    labels: np.ndarray,
    tfidf_vectorizer,
    k: int,
    n_words: int = 10,
) -> dict[int, list[str]]:

    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    keywords = {}

    for cluster_id in range(k):
        mask   = labels == cluster_id
        subset = tfidf_matrix[mask]

        if subset.shape[0] == 0:
            keywords[cluster_id] = []
            continue

        mean_tfidf = subset.mean(axis=0)

        # Handle sparse properly
        if hasattr(mean_tfidf, "toarray"):
            mean_tfidf = mean_tfidf.toarray().flatten()
        else:
            mean_tfidf = np.array(mean_tfidf).flatten()

        # 🔥 Safe sorting
        top_idx = np.argsort(mean_tfidf)[::-1][:n_words]

        # 🔒 Prevent out-of-bounds crash
        top_idx = top_idx[top_idx < len(feature_names)]

        keywords[cluster_id] = feature_names[top_idx].tolist()

    return keywords


# ── Cluster-wise sentiment analysis ──────────────────────────────────────────
def cluster_sentiment_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """
    Given a DataFrame with 'predicted_label' column and cluster labels,
    produce a per-cluster sentiment distribution table.

    Returns
    -------
    DataFrame with columns:
        cluster_id, size, pct_negative, pct_neutral, pct_positive, dominant_sentiment
    """
    df = df.copy()
    df["cluster"] = labels

    rows = []
    for cid in range(k):
        sub    = df[df["cluster"] == cid]
        n      = len(sub)
        counts = sub["predicted_label"].value_counts()

        n_neg  = counts.get(0, 0)
        n_neu  = counts.get(1, 0)
        n_pos  = counts.get(2, 0)

        pct_neg = round(n_neg / n * 100, 1) if n else 0
        pct_neu = round(n_neu / n * 100, 1) if n else 0
        pct_pos = round(n_pos / n * 100, 1) if n else 0

        dominant_id  = counts.idxmax() if n else 1
        dominant_str = LABEL_MAP.get(dominant_id, "Neutral")

        rows.append({
            "Cluster":            cid,
            "Size":               n,
            "% Negative":         pct_neg,
            "% Neutral":          pct_neu,
            "% Positive":         pct_pos,
            "Dominant Sentiment": dominant_str,
        })

    return pd.DataFrame(rows)


# ── Full pipeline wrapper ─────────────────────────────────────────────────────
def run_clustering(
    df: pd.DataFrame,
    tfidf_matrix,
    tfidf_vectorizer,
    k: int = DEFAULT_K,
) -> dict:
    """
    End-to-end clustering pipeline.

    Parameters
    ----------
    df               : DataFrame that already has a 'predicted_label' column
    tfidf_matrix     : sparse TF-IDF matrix aligned with df rows
    tfidf_vectorizer : fitted TfidfVectorizer

    Returns
    -------
    dict with: labels, km_model, summary_df, keywords
    """
    labels, km, lsa, svd = cluster_tweets(tfidf_matrix, k=k)
    kw      = top_keywords(tfidf_matrix, labels, tfidf_vectorizer, k=k)
    summary = cluster_sentiment_summary(df, labels, k=k)

    return {
        "labels":   labels,
        "km_model": km,
        "lsa":      lsa,
        "summary":  summary,
        "keywords": kw,
    }