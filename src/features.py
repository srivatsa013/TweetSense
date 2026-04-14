"""
src/features.py
---------------
Feature Engineering Module  (SDD §3.2)

Extracts:
  • tweet_length        – character count of original tweet
  • word_count          – word count
  • hashtag_count       – number of hashtags
  • mention_count       – number of @mentions
  • exclamation_count   – number of '!' characters
  • question_count      – number of '?' characters
  • uppercase_ratio     – fraction of alphabetic chars that are uppercase
  • emoji_sentiment     – compound emoji sentiment score  (-1 … +1)
  • emoji_count         – total emoji characters found

These are stacked with TF-IDF vectors in the final feature matrix.
"""

import re
import numpy as np
import pandas as pd

try:
    import emoji as emoji_lib
    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False


# ── Emoji sentiment lexicon ───────────────────────────────────────────────────
# Curated scores for the most common emojis found in tweets.
# Score range: -1 (very negative) → 0 (neutral) → +1 (very positive)
EMOJI_SENTIMENT: dict[str, float] = {
    # Strongly positive
    "😍": 1.0, "🥰": 1.0, "😁": 0.9, "😊": 0.9, "🤗": 0.9,
    "😄": 0.85, "😃": 0.85, "🎉": 0.9, "🥳": 0.9, "❤️": 0.95,
    "💕": 0.9,  "💯": 0.85, "👏": 0.8, "🙌": 0.85, "✨": 0.75,
    "😂": 0.7,  "🤣": 0.7,  "😆": 0.7, "😎": 0.75, "🔥": 0.7,
    "💪": 0.75, "👍": 0.8,  "🌟": 0.8, "😀": 0.85, "🫶": 0.9,
    "💙": 0.85, "💚": 0.8,  "🧡": 0.8, "💛": 0.8,  "💜": 0.8,
    "🎊": 0.85, "🥂": 0.8,  "🍾": 0.8, "🌈": 0.75, "☀️": 0.7,
    "😋": 0.75, "🤩": 0.9,  "😇": 0.85,"💖": 0.9,  "🤍": 0.8,
    # Mildly positive
    "🙂": 0.5,  "😏": 0.3,  "😌": 0.4, "🤔": 0.1, "🫠": 0.1,
    "😮": 0.2,  "🤭": 0.4,  "😉": 0.5, "👌": 0.6,  "💫": 0.5,
    # Neutral
    "😐": 0.0,  "😑": 0.0,  "🤷": 0.0, "🫡": 0.0,
    # Mildly negative
    "😕": -0.4, "🙁": -0.5, "😟": -0.5,"😬": -0.3, "😒": -0.4,
    "😤": -0.5, "🫤": -0.3, "😔": -0.5,"🥲": -0.3, "😥": -0.5,
    # Strongly negative
    "😢": -0.85,"😭": -0.9, "😠": -0.8,"😡": -0.9, "🤬": -0.95,
    "💔": -0.9, "😖": -0.8, "😣": -0.75,"😩": -0.8,"😫": -0.8,
    "🤮": -0.9, "🤢": -0.85,"💀": -0.7, "☠️": -0.75,"😰": -0.7,
    "😨": -0.65,"😱": -0.65,"😳": -0.5, "👎": -0.8, "🖕": -0.95,
}


def _emoji_sentiment_score(text: str) -> tuple[float, int]:
    """
    Compute an aggregate emoji sentiment score for *text*.

    Returns
    -------
    (score, count)
        score : mean sentiment of all scored emojis, or 0.0 if none found
        count : total number of emoji characters found
    """
    scores = []
    count  = 0

    if _EMOJI_AVAILABLE:
        # Use the `emoji` library to extract all emoji from text
        emoji_list = [e["emoji"] for e in emoji_lib.emoji_list(text)]
    else:
        # Fallback: match characters in our lexicon
        emoji_list = [c for c in text if c in EMOJI_SENTIMENT]

    for em in emoji_list:
        count += 1
        if em in EMOJI_SENTIMENT:
            scores.append(EMOJI_SENTIMENT[em])

    score = float(np.mean(scores)) if scores else 0.0
    return score, count


# ── Core feature extraction ───────────────────────────────────────────────────
def extract_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Given a DataFrame with a raw-text column, return a new DataFrame
    containing all engineered numeric features.

    Parameters
    ----------
    df       : input DataFrame (must contain *text_col*)
    text_col : column name of the raw tweet text

    Returns
    -------
    pd.DataFrame with columns:
        tweet_length, word_count, hashtag_count, mention_count,
        exclamation_count, question_count, uppercase_ratio,
        emoji_sentiment, emoji_count
    """
    texts = df[text_col].astype(str)

    def _safe(fn, fallback=0):
        def wrapped(t):
            try:
                return fn(t)
            except Exception:
                return fallback
        return wrapped

    features = pd.DataFrame()

    features["tweet_length"]      = texts.apply(len)
    features["word_count"]        = texts.apply(lambda t: len(t.split()))
    features["hashtag_count"]     = texts.apply(lambda t: len(re.findall(r"#\w+", t)))
    features["mention_count"]     = texts.apply(lambda t: len(re.findall(r"@\w+", t)))
    features["exclamation_count"] = texts.apply(lambda t: t.count("!"))
    features["question_count"]    = texts.apply(lambda t: t.count("?"))
    features["uppercase_ratio"]   = texts.apply(
        lambda t: (
            sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1)
        )
    )

    emoji_results = texts.apply(_emoji_sentiment_score)
    features["emoji_sentiment"] = emoji_results.apply(lambda x: x[0])
    features["emoji_count"]     = emoji_results.apply(lambda x: x[1])

    return features.reset_index(drop=True)