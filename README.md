# 🐦 TweetSense — Twitter Sentiment Analytics

An end-to-end Machine Learning pipeline for **Twitter sentiment analysis**, built with modular design principles aligned with SRS and SDD specifications.
Combines **NLP, supervised learning, and unsupervised clustering** with an interactive Streamlit dashboard.

---

## 🚀 Features

- 🔍 Real-time tweet sentiment prediction  
- 🧠 SVM / Logistic Regression classification models  
- 📊 Interactive Streamlit dashboard  
- 🗂️ K-Means clustering with keyword extraction  
- 😊 Emoji-aware sentiment scoring  
- ⚡ Optimized pipeline using TF-IDF + LSA 

---

## 📁 Project Structure

```
tweet_sentiment/
├── app.py                  # Streamlit dashboard (main entry point)
├── download_data.py        # One-time dataset downloader
├── requirements.txt        # Python dependencies
├── data/                   # Auto-generated (git-ignored)
│   ├── tweets_sample.csv
│   └── tweets.csv
├── models/                 # Saved model artefacts
│   ├── classifier.pkl
│   └── tfidf.pkl
└── src/
    ├── preprocess.py       # Text cleaning & tokenisation
    ├── features.py         # Feature engineering + emoji scoring
    ├── model.py            # Model training (SVM / LR)
    └── cluster.py          # K-Means clustering + analysis
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset *(one-time setup)*
```bash
python download_data.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🔬 Machine Learning Pipeline

```
Raw Tweets
    ↓
Preprocessing
    → Lowercasing, URL removal, tokenisation, stopword removal
    ↓
Feature Engineering
    → Tweet length, hashtag count, punctuation density, emoji sentiment score
    ↓
TF-IDF Vectorisation
    → 30,000 features · bigrams · sublinear TF scaling
    ↓
Supervised Learning
    → Linear SVM (calibrated) / Logistic Regression
    ↓
Sentiment Classification
    → Negative · Neutral · Positive
    ↓
Unsupervised Clustering
    → LSA (Truncated SVD) → MiniBatch K-Means
    ↓
Cluster Insights
    → Top keywords + sentiment distribution per cluster
```

---

## 🏷️ Sentiment Classes

The original Sentiment140 dataset contains binary labels:

| Original Label | Meaning   |
|---------------|-----------|
| `0`           | Negative  |
| `4`           | Positive  |

This project extends to **3-class classification**:

| Class | Label     |
|-------|-----------|
| `0`   | Negative  |
| `1`   | Neutral   |
| `2`   | Positive  |

** Neutral classification strategy:**
```
if max(predict_proba) < 0.62 → classify as Neutral
```
This threshold avoids overconfident predictions and improves real-world usability.

---

## 📊 Dashboard Overview

| Tab | Functionality |
|-----|--------------|
| 📂 Data | Load and explore the dataset |
| 🏋️ Train | Train model and evaluate performance |
| 🔍 Predict | Live tweet sentiment prediction |
| 🗂️ Clusters | K-Means clustering with topic insights |
| 📊 Analytics | Visualisations and sentiment distributions |

---

## 📦 Dataset

- **Sentiment140** — ~1.6 million labelled tweets
- Automatically downloaded via `download_data.py`
- Source: [Stanford NLP — Sentiment140](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

> **Note:** Dataset files are excluded from this repository due to size. Run `python download_data.py` to fetch and prepare the dataset locally.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | Streamlit |
| ML / NLP | Scikit-learn, TF-IDF, LSA |
| Data | Pandas, NumPy |
| Serialisation | Pickle |
| Language | Python 3.x |

---

## ✨ Future Improvements

- Deep learning models (LSTM / BERT fine-tuning)
- Real-time Twitter/X API integration
- Advanced topic modelling with LDA
- Deployment via Streamlit Cloud or Docker

---

## 📚 References

Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* Stanford University Technical Report.

---

## 👨‍💻 Author

**Srivatsa Singaraju**
