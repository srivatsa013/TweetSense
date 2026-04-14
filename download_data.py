"""
download_data.py
----------------
Downloads and prepares the Sentiment140 dataset (1.6M tweets).
Run this once before launching the app:  python download_data.py
"""

import os
import zipfile
import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
ZIP_PATH   = os.path.join(DATA_DIR, "sentiment140.zip")
CSV_PATH   = os.path.join(DATA_DIR, "tweets.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "tweets_sample.csv")

DOWNLOAD_URL = (
    "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
)

COLUMNS = ["target", "id", "date", "flag", "user", "text"]
SAMPLE_SIZE = 50_000   # rows used by default in the app (fast + representative)


# ── Helpers ───────────────────────────────────────────────────────────────────
def download_zip(url: str, dest: str) -> None:
    """Stream-download a ZIP file with a progress bar."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset from:\n  {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%  ({downloaded >> 20} / {total >> 20} MB)", end="")
    print("\nDownload complete.")


def extract_zip(zip_path: str, dest_dir: str) -> str:
    """Extract zip and return path to the training CSV."""
    print("Extracting archive …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
        names = z.namelist()
    # The training file is the one that contains 'training'
    training_file = next(
        (os.path.join(dest_dir, n) for n in names if "training" in n.lower()), None
    )
    if training_file is None:
        raise FileNotFoundError("Could not find training CSV inside the zip.")
    print(f"Extracted: {training_file}")
    return training_file


def prepare_dataset(raw_csv: str) -> None:
    """Read raw Sentiment140 CSV, clean up columns, save tidy version + sample."""
    print("Preparing dataset …")
    df = pd.read_csv(
        raw_csv,
        encoding="latin-1",
        header=None,
        names=COLUMNS,
        usecols=["target", "text"],
    )

    # Map binary labels:  0 → Negative (0),  4 → Positive (2)
    # Neutral (1) is derived later at inference time via model confidence.
    df["label"] = df["target"].map({0: 0, 4: 2})
    df = df[["text", "label"]].dropna()

    # Save full cleaned CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Full dataset saved → {CSV_PATH}  ({len(df):,} rows)")

    # Balanced sample  (equal pos / neg)
    half = SAMPLE_SIZE // 2
    sample = pd.concat([
        df[df["label"] == 0].sample(n=half, random_state=42),
        df[df["label"] == 2].sample(n=half, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Balanced sample saved → {SAMPLE_PATH}  ({len(sample):,} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if os.path.exists(SAMPLE_PATH):
        print(f"Sample already exists at {SAMPLE_PATH}. Nothing to do.")
        print("Delete it and re-run to force a fresh download.")
        return

    # 🧹 If zip exists but is corrupted → delete it
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.testzip()  # checks integrity
        except zipfile.BadZipFile:
            print("⚠️ Corrupted ZIP detected. Deleting and re-downloading...")
            os.remove(ZIP_PATH)

    # 📥 Download if not present (or just deleted)
    if not os.path.exists(ZIP_PATH):
        download_zip(DOWNLOAD_URL, ZIP_PATH)

    raw_csv = extract_zip(ZIP_PATH, DATA_DIR)
    prepare_dataset(raw_csv)

    print("\n✅ Dataset ready. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()