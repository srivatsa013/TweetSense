"""
app.py
------
Streamlit Dashboard  (SDD §5.1)

Tabs:
  1. 📂 Data         – load & preview dataset
  2. 🏋️ Train        – train SVM or Logistic Regression
  3. 🔍 Predict      – live single-tweet prediction
  4. 🗂️ Clusters     – K-Means + cluster-wise analysis
  5. 📊 Analytics    – visualisations (sentiment distribution, word clouds)

Run:
    streamlit run app.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="TweetSense · Sentiment Analytics",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.5px; }

  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 1rem 1.2rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #0a0f1e;
      border-right: 1px solid #1e293b;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
      gap: 4px;
      background: #0f172a;
      padding: 4px;
      border-radius: 10px;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      color: #94a3b8;
      border-radius: 8px;
      font-family: 'Space Mono', monospace;
      font-size: 0.78rem;
  }
  .stTabs [aria-selected="true"] {
      background: #1e3a5f !important;
      color: #38bdf8 !important;
  }

  /* Status badges */
  .badge-pos  { background:#14532d; color:#86efac; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
  .badge-neg  { background:#450a0a; color:#fca5a5; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }
  .badge-neu  { background:#422006; color:#fcd34d; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.85rem; }

  /* Info boxes */
  .info-box {
      background: #0f172a;
      border-left: 3px solid #38bdf8;
      padding: 0.8rem 1rem;
      border-radius: 0 8px 8px 0;
      margin: 0.5rem 0;
      font-size: 0.9rem;
      color: #cbd5e1;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "df":             None,
    "clf":            None,
    "tfidf":          None,
    "metrics":        None,
    "pred_df":        None,
    "cluster_result": None,
    "tfidf_matrix":   None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐦 TweetSense")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    model_choice = st.radio("ML Model", ["SVM (LinearSVC)", "Logistic Regression"])
    n_clusters   = st.slider("K-Means clusters", min_value=2, max_value=10, value=5)
    sample_size  = st.slider("Dataset sample size", 5_000, 50_000, 20_000, step=5_000)

    st.markdown("---")
    st.markdown(
        "<div class='info-box'>Dataset: Sentiment140<br>"
        "Labels: Neg / Neutral / Pos<br>"
        "Neutral via confidence threshold</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Built following SRS + SDD pipeline")


# ── Helpers ────────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "Negative": "#ef4444",
    "Neutral":  "#f59e0b",
    "Positive": "#22c55e",
}

def sentiment_badge(label: str) -> str:
    cls = {"Positive": "badge-pos", "Negative": "badge-neg"}.get(label, "badge-neu")
    return f"<span class='{cls}'>{label}</span>"

@st.cache_data(show_spinner=False)
def load_sample(path: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    half = n // 2
    neg = df[df["label"] == 0].sample(min(half, len(df[df["label"]==0])), random_state=42)
    pos = df[df["label"] == 2].sample(min(half, len(df[df["label"]==2])), random_state=42)
    return pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📂 Data", "🏋️ Train", "🔍 Predict", "🗂️ Clusters", "📊 Analytics"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  ·  Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Dataset Loader")

    SAMPLE_PATH = os.path.join("data", "tweets_sample.csv")
    CSV_PATH    = os.path.join("data", "tweets.csv")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("📥 Load Dataset", type="primary", use_container_width=True):
            if not os.path.exists(SAMPLE_PATH):
                st.error(
                    "Dataset not found. Run `python download_data.py` first "
                    "to download Sentiment140."
                )
            else:
                with st.spinner("Loading and preprocessing …"):
                    sys.path.insert(0, os.path.dirname(__file__))
                    from src.preprocess import preprocess_dataframe

                    raw = load_sample(SAMPLE_PATH, sample_size)
                    df  = preprocess_dataframe(raw, text_col="text")
                    st.session_state.df = df
                st.success(f"✅ Loaded {len(df):,} tweets")

    with col2:
        st.markdown(
            "<div class='info-box'>Need data? Run:<br>"
            "<code>python download_data.py</code></div>",
            unsafe_allow_html=True,
        )

    if st.session_state.df is not None:
        df = st.session_state.df

        # KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tweets",  f"{len(df):,}")
        m2.metric("Negative",      f"{(df.label==0).sum():,}")
        m3.metric("Positive",      f"{(df.label==2).sum():,}")
        m4.metric("Vocab (approx)", "—")

        st.markdown("#### 🔍 Sample Tweets")
        preview = df[["text", "clean_text", "label"]].head(10).copy()
        preview["Sentiment"] = preview["label"].map({0:"Negative", 2:"Positive"})
        st.dataframe(
            preview[["text", "clean_text", "Sentiment"]],
            use_container_width=True,
            height=320,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  ·  Train
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Model Training")

    if st.session_state.df is None:
        st.info("⬆️ Load the dataset first (Tab 1)")
    else:
        model_type = "svm" if "SVM" in model_choice else "lr"

        col_btn, col_note = st.columns([1, 2])
        with col_btn:
            train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)
        with col_note:
            st.markdown(
                "<div class='info-box'>Pipeline: Preprocessing → Feature Engineering "
                "→ TF-IDF (30k) → " + model_choice.split(" ")[0] + "</div>",
                unsafe_allow_html=True,
            )

        if train_btn:
            from src.features import extract_features
            from src.model import train, build_feature_matrix

            df = st.session_state.df

            with st.spinner("Extracting features …"):
                eng = extract_features(df, text_col="text")
                df_feat = pd.concat([df.reset_index(drop=True), eng], axis=1)

            with st.spinner(f"Training {model_choice} …"):
                t0     = time.time()
                result = train(df_feat, model_type=model_type)
                elapsed = time.time() - t0

            st.session_state.clf         = result["model"]
            st.session_state.tfidf       = result["tfidf"]
            st.session_state.metrics     = result["metrics"]

            # Build full TF-IDF matrix for clustering
            tfidf_mat, _ = build_feature_matrix(
                df["clean_text"], eng, result["tfidf"], fit_tfidf=False
            )
            # Store only TF-IDF part (first 30k cols) for clustering
            st.session_state.tfidf_matrix = tfidf_mat[:, :30_000]

            # Apply predictions to whole dataset
            from src.model import _apply_neutral
            probas  = result["model"].predict_proba(tfidf_mat)
            raw_pred = result["model"].predict(tfidf_mat)
            pred3    = _apply_neutral(probas, raw_pred)
            df["predicted_label"] = pred3
            st.session_state.pred_df = df

            m = result["metrics"]
            st.success(f"✅ Training complete in {elapsed:.1f}s")

            # Metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",    f"{m['accuracy']}%")
            c2.metric("Train rows",  f"{m['n_train']:,}")
            c3.metric("Test rows",   f"{m['n_test']:,}")
            c4.metric("Model",       m['model_type'])

            # Confusion matrix
            cm   = m["confusion_matrix"]
            fig  = go.Figure(go.Heatmap(
                z=cm[::-1],
                x=["Pred: Neg", "Pred: Pos"],
                y=["True: Pos", "True: Neg"],
                colorscale="Blues",
                showscale=True,
                text=cm[::-1],
                texttemplate="%{text}",
                textfont={"size": 18},
            ))
            fig.update_layout(
                title="Confusion Matrix (binary ground truth)",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#cbd5e1",
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Raw report
            with st.expander("Full classification report"):
                st.code(m["classification_report"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  ·  Predict
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Live Tweet Predictor")

    if st.session_state.clf is None:
        st.info("⬆️ Train a model first (Tab 2)")
    else:
        tweet_input = st.text_area(
            "Enter a tweet:",
            placeholder="e.g.  Just had the best coffee ☕🎉 — feeling fantastic today!!",
            height=100,
        )

        if st.button("🔮 Analyse Sentiment", type="primary"):
            if tweet_input.strip():
                from src.model import predict_single
                res = predict_single(
                    tweet_input,
                    st.session_state.clf,
                    st.session_state.tfidf,
                )

                col_r, col_p = st.columns([1, 2])
                with col_r:
                    st.markdown("#### Result")
                    st.markdown(sentiment_badge(res["label_str"]), unsafe_allow_html=True)
                    st.metric("Confidence", f"{res['confidence']}%")

                with col_p:
                    st.markdown("#### Probability Breakdown")
                    prob_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Positive"],
                        "Probability (%)": [res["probas"]["Negative"], res["probas"]["Positive"]],
                    })
                    fig = px.bar(
                        prob_df, x="Sentiment", y="Probability (%)",
                        color="Sentiment",
                        color_discrete_map={"Negative":"#ef4444","Positive":"#22c55e"},
                        range_y=[0, 100],
                    )
                    fig.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font_color="#cbd5e1", showlegend=False, height=260,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  ·  Clusters
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Tweet Clustering  (K-Means)")

    if st.session_state.pred_df is None:
        st.info("⬆️ Train a model first (Tab 2)")
    else:
        if st.button(f"🔵 Run K-Means  (k={n_clusters})", type="primary"):
            from src.cluster import run_clustering

            with st.spinner("Clustering tweets …"):
                result = run_clustering(
                    st.session_state.pred_df,
                    st.session_state.tfidf_matrix,
                    st.session_state.tfidf,
                    k=n_clusters,
                )
                st.session_state.cluster_result = result

        if st.session_state.cluster_result is not None:
            cr      = st.session_state.cluster_result
            summary = cr["summary"]
            kw      = cr["keywords"]

            st.markdown("#### Cluster Summary")
            st.dataframe(summary, use_container_width=True)

            # Stacked bar chart: sentiment per cluster
            fig = go.Figure()
            for sentiment, col in [("% Negative","#ef4444"),("% Neutral","#f59e0b"),("% Positive","#22c55e")]:
                fig.add_trace(go.Bar(
                    name=sentiment.replace("% ",""),
                    x=summary["Cluster"].astype(str),
                    y=summary[sentiment],
                    marker_color=col,
                ))
            fig.update_layout(
                barmode="stack",
                title="Sentiment Distribution per Cluster",
                xaxis_title="Cluster",
                yaxis_title="Percentage (%)",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#cbd5e1",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top keywords per cluster
            st.markdown("#### 🔑 Top Keywords per Cluster")
            kw_cols = st.columns(min(n_clusters, 5))
            for cid, col in enumerate(kw_cols):
                with col:
                    dominant = summary.loc[summary["Cluster"]==cid, "Dominant Sentiment"].values[0]
                    badge    = sentiment_badge(dominant)
                    st.markdown(f"**Cluster {cid}** {badge}", unsafe_allow_html=True)
                    for word in kw.get(cid, [])[:8]:
                        st.markdown(f"• {word}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  ·  Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Analytics & Insights")

    if st.session_state.pred_df is None:
        st.info("⬆️ Train a model first (Tab 2)")
    else:
        df = st.session_state.pred_df

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df["Sentiment"] = df["predicted_label"].map(label_map)

        # ── Row 1: Sentiment distribution ──
        col1, col2 = st.columns(2)

        with col1:
            counts = df["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            fig = px.pie(
                counts, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.45,
                title="Overall Sentiment Distribution",
            )
            fig.update_layout(
                paper_bgcolor="#0f172a", font_color="#cbd5e1", height=340
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                counts, x="Sentiment", y="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Tweet Counts by Sentiment",
            )
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#cbd5e1", showlegend=False, height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Row 2: Engineered feature distributions ──
        st.markdown("#### Feature Distributions by Sentiment")

        from src.features import extract_features
        eng = extract_features(df, text_col="text")
        eng["Sentiment"] = df["Sentiment"].values

        feat_col = st.selectbox(
            "Select feature:",
            ["tweet_length", "hashtag_count", "exclamation_count",
             "emoji_sentiment", "emoji_count", "word_count"],
        )

        fig = px.box(
            eng, x="Sentiment", y=feat_col,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title=f"{feat_col} distribution per sentiment class",
            points="outliers",
        )
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#cbd5e1", showlegend=False, height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Row 3: Word Cloud ──
        st.markdown("#### ☁️ Word Cloud")

        wc_sentiment = st.radio(
            "Sentiment class:", ["Positive", "Negative", "Neutral"], horizontal=True
        )
        wc_label = {"Positive": 2, "Negative": 0, "Neutral": 1}[wc_sentiment]
        wc_texts = df[df["predicted_label"] == wc_label]["clean_text"]

        if len(wc_texts) > 0:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                wc_color = {"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#f59e0b"}[wc_sentiment]
                wc = WordCloud(
                    width=900, height=350,
                    background_color="#0f172a",
                    color_func=lambda *a, **k: wc_color,
                    max_words=120,
                    collocations=False,
                ).generate(" ".join(wc_texts.sample(min(2000, len(wc_texts)), random_state=42)))

                fig, ax = plt.subplots(figsize=(11, 4))
                fig.patch.set_facecolor("#0f172a")
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close()
            except ImportError:
                st.warning("Install `wordcloud` to enable this visualisation:  `pip install wordcloud`")
        else:
            st.info("No tweets in this class yet.")