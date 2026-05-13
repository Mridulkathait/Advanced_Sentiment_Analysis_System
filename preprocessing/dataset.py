"""Dataset loading, cleaning, balancing, and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from preprocessing.pipeline import TextPreprocessor
from utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


def normalize_label(raw: str, settings: Settings) -> Optional[str]:
    v = str(raw).strip().lower()
    if v in {x.lower() for x in settings.positive_labels}:
        return "positive"
    if v in {x.lower() for x in settings.negative_labels}:
        return "negative"
    if v in {x.lower() for x in settings.neutral_labels}:
        return "neutral"
    return None


def load_reviews_csv(
    path: str | Path,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    settings = settings or get_settings()
    text_col = text_col or settings.text_column
    label_col = label_col or settings.label_column
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expected columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}")
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label_raw"]
    df["label"] = df["label_raw"].map(lambda x: normalize_label(str(x), settings))
    before = len(df)
    df = df.dropna(subset=["label"])
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].astype(str).str.len() > 0]
    logger.info("Loaded %s rows, kept %s after cleaning (dropped %s)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def balance_classes(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Random oversample minority classes to match majority count."""
    parts = []
    max_n = df["label"].value_counts().max()
    for lab, grp in df.groupby("label"):
        if len(grp) < max_n:
            grp = resample(grp, replace=True, n_samples=max_n, random_state=random_state)
        parts.append(grp)
    out = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out


def build_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])


def preprocess_dataframe(df: pd.DataFrame, preprocessor: TextPreprocessor) -> pd.DataFrame:
    out = df.copy()
    out["processed"] = preprocessor.transform_series(out["text"])
    out = out[out["processed"].str.len() > 0]
    return out.reset_index(drop=True)
