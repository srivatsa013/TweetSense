"""
src/preprocess.py
-----------------
Preprocessing Module  (SDD §3.1)

Cleans raw tweet text:
  - Lowercasing
  - Remove URLs, mentions, hashtag symbols, special characters
  - Tokenization
  - Stopword removal
  - (Emoji characters are preserved so features.py can score them)
"""

import re
import nltk
import unicodedata

# Download NLTK resources on first run
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────
STOP_WORDS  = set(stopwords.words("english"))
_TOKENIZER  = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Negation words — keep them even if they're stopwords
NEGATIONS = {"not", "no", "never", "neither", "nor", "nothing", "nobody",
             "nowhere", "cannot", "cant", "wont", "dont", "doesnt", "didnt",
             "isnt", "wasnt", "wouldnt", "shouldnt", "couldnt", "hadnt",
             "hasnt", "havent"}

EFFECTIVE_STOP = STOP_WORDS - NEGATIONS


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_emoji(char: str) -> bool:
    """Return True if the character is an emoji / pictographic symbol."""
    cat = unicodedata.category(char)
    cp  = ord(char)
    return cat in ("So", "Sm") or (0x1F300 <= cp <= 0x1FFFF) or (0x2600 <= cp <= 0x26FF)


def extract_emojis(text: str) -> list[str]:
    """Return list of emoji characters found in *text*."""
    return [c for c in text if _is_emoji(c)]


def clean_tweet(text: str, keep_emojis: bool = True) -> str:
    """
    Full cleaning pipeline for a single tweet.

    Parameters
    ----------
    text        : raw tweet string
    keep_emojis : if True, emoji characters are preserved in output
                  (features.py will score them separately)

    Returns
    -------
    Cleaned, lowercased, tokenized string joined by spaces.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 2. Remove @mentions  (TweetTokenizer does this too, belt-and-suspenders)
    text = re.sub(r"@\w+", "", text)

    # 3. Optionally remove emojis before tokenisation so they survive intact
    emoji_chars = extract_emojis(text) if keep_emojis else []

    # 4. Remove hashtag symbol but keep the word  (#happy → happy)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Tokenise (handles contractions, slang elongation, etc.)
    tokens = _TOKENIZER.tokenize(text)

    # 6. Keep only alphabetic tokens + remove stopwords
    tokens = [t for t in tokens if t.isalpha() and t not in EFFECTIVE_STOP]

    # 7. Re-attach emojis so the vectoriser sees the cleaned text
    cleaned = " ".join(tokens)
    if keep_emojis and emoji_chars:
        cleaned = cleaned + " " + " ".join(emoji_chars)

    return cleaned.strip()


def preprocess_dataframe(df, text_col: str = "text") -> "pd.DataFrame":
    """
    Apply clean_tweet to an entire DataFrame column.

    Adds a new column  'clean_text'  to *df* (in-place copy returned).
    """
    import pandas as pd
    df = df.copy()
    df["clean_text"] = df[text_col].astype(str).apply(clean_tweet)
    # Drop rows where cleaning produced empty strings
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    return df