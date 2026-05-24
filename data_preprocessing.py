"""
data_preprocessing.py
=====================
Loads and cleans the anime dataset for use in all recommendation models.
"""

import pandas as pd
import numpy as np
import re
import os

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "anime-dataset-2023.csv"
PROCESSED_DATA_PATH = "processed_anime.csv"


# ══════════════════════════════════════════════════════════════════════════════
def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please place 'anime-dataset-2023.csv' in the project root."
        )
    df = pd.read_csv(path)
    print(f"[load] {len(df):,} rows, {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def clean_score(val) -> float:
    """Convert score column to float; map 'UNKNOWN' → NaN."""
    if isinstance(val, str) and val.strip().upper() == "UNKNOWN":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def clean_episodes(val) -> float:
    """Convert episodes column to float; handle 'UNKNOWN'."""
    if isinstance(val, str) and val.strip().upper() == "UNKNOWN":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def clean_duration_minutes(val) -> float:
    """
    Parse duration strings like '24 min per ep', '1 hr 55 min', 'Unknown'
    and return total minutes as a float.
    """
    if not isinstance(val, str):
        return np.nan
    val = val.lower().strip()
    if "unknown" in val:
        return np.nan

    hours, minutes = 0, 0
    hr_match = re.search(r"(\d+)\s*hr", val)
    min_match = re.search(r"(\d+)\s*min", val)
    if hr_match:
        hours = int(hr_match.group(1))
    if min_match:
        minutes = int(min_match.group(1))
    total = hours * 60 + minutes
    return float(total) if total > 0 else np.nan


def clean_rank(val) -> float:
    """Convert rank to float; handle 'UNKNOWN'."""
    if isinstance(val, str) and val.strip().upper() == "UNKNOWN":
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def clean_scored_by(val) -> float:
    """Convert 'Scored By' to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop duplicates
      2. Standardise string unknowns
      3. Fix numeric columns
      4. Fill remaining NaNs with sensible defaults
      5. Create helper columns used by the recommenders
    Returns a cleaned DataFrame.
    """
    print("[preprocess] starting ...")
    df = df.copy()

    # 1. drop exact duplicates
    before = len(df)
    df.drop_duplicates(subset="anime_id", keep="first", inplace=True)
    print(f"  removed {before - len(df)} duplicate anime_id rows")

    # 2. replace literal 'UNKNOWN' strings in text columns with NaN
    text_cols = ["Genres", "Synopsis", "Type", "Status", "Source", "Rating",
                 "Premiered", "Aired", "Producers", "Licensors", "Studios",
                 "English name", "Other name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace({"UNKNOWN": np.nan, "Unknown": np.nan})

    # 3. fix numeric columns
    df["Score"]      = df["Score"].apply(clean_score)
    df["Episodes"]   = df["Episodes"].apply(clean_episodes)
    df["Rank"]       = df["Rank"].apply(clean_rank)
    df["Scored By"]  = df["Scored By"].apply(clean_scored_by)
    df["Duration_min"] = df["Duration"].apply(clean_duration_minutes)

    # 4. fill NaNs
    df["Score"]       = df["Score"].fillna(df["Score"].median())
    df["Episodes"]    = df["Episodes"].fillna(df["Episodes"].median())
    df["Rank"]        = df["Rank"].fillna(df["Rank"].max() + 1)   # worst rank
    df["Scored By"]   = df["Scored By"].fillna(0)
    df["Duration_min"] = df["Duration_min"].fillna(df["Duration_min"].median())
    df["Genres"]      = df["Genres"].fillna("Unknown")
    df["Synopsis"]    = df["Synopsis"].fillna("")
    df["Type"]        = df["Type"].fillna("Unknown")
    df["Studios"]     = df["Studios"].fillna("Unknown")
    df["Rating"]      = df["Rating"].fillna("Unknown")
    df["Source"]      = df["Source"].fillna("Unknown")

    # 5. helper columns
    #    genre list  e.g. "Action, Sci-Fi" → ["Action", "Sci-Fi"]
    df["genre_list"] = df["Genres"].apply(
        lambda g: [x.strip() for x in str(g).split(",") if x.strip() and x.strip() != "Unknown"]
    )

    #    combined text feature for TF-IDF / synopsis-based model
    df["text_features"] = (
        df["Genres"].fillna("") + " " +
        df["Type"].fillna("") + " " +
        df["Studios"].fillna("") + " " +
        df["Source"].fillna("") + " " +
        df["Rating"].fillna("") + " " +
        df["Synopsis"].fillna("")
    )

    #    popularity score (log-scaled members)
    df["popularity_score"] = np.log1p(df["Members"])

    print(f"[preprocess] done - {len(df):,} rows remaining")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"[save] processed data -> '{path}'")


def load_processed(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned CSV produced by save_processed()."""
    df = pd.read_csv(path)
    # restore list column
    import ast
    df["genre_list"] = df["genre_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    raw = load_raw_data()
    clean = preprocess(raw)
    save_processed(clean)
    print(clean[["Name", "Score", "Episodes", "Genres", "Duration_min"]].head())
