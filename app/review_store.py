"""SQLite persistence for analyzed reviews (session-long and cross-session insights)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.config import Settings, get_settings


def _db_path(settings: Settings) -> Path:
    return settings.data_dir / "review_history.sqlite3"


def _connect(settings: Settings) -> sqlite3.Connection:
    path = _db_path(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    conn = _connect(settings)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text TEXT NOT NULL,
                model TEXT,
                overall_label TEXT,
                positive_pct REAL,
                negative_pct REAL,
                neutral_pct REAL,
                aspects_json TEXT,
                scores_json TEXT
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_created ON reviews(created_at);")
        conn.commit()
    finally:
        conn.close()


def save_review(
    text: str,
    model: str,
    overall_label: str,
    positive_pct: float,
    negative_pct: float,
    neutral_pct: float,
    aspects: List[Dict[str, Any]],
    scores: Dict[str, Any],
    settings: Optional[Settings] = None,
) -> int:
    settings = settings or get_settings()
    init_db(settings)
    conn = _connect(settings)
    try:
        cur = conn.execute(
            """
            INSERT INTO reviews (
                created_at, text, model, overall_label,
                positive_pct, negative_pct, neutral_pct,
                aspects_json, scores_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                text[:8000],
                model,
                overall_label,
                float(positive_pct),
                float(negative_pct),
                float(neutral_pct),
                json.dumps(aspects, ensure_ascii=False),
                json.dumps(scores, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)
    finally:
        conn.close()


def load_reviews_df(limit: Optional[int] = None, settings: Optional[Settings] = None) -> pd.DataFrame:
    settings = settings or get_settings()
    if not _db_path(settings).exists():
        return pd.DataFrame()
    init_db(settings)
    conn = _connect(settings)
    try:
        q = "SELECT * FROM reviews ORDER BY id DESC"
        if limit is not None:
            return pd.read_sql_query(q + " LIMIT ?", conn, params=(int(limit),))
        return pd.read_sql_query(q, conn)
    finally:
        conn.close()


def clear_all(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not _db_path(settings).exists():
        return
    conn = _connect(settings)
    try:
        conn.execute("DELETE FROM reviews;")
        conn.commit()
    finally:
        conn.close()


def aggregate_insights(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Summary stats for dashboards."""
    df = load_reviews_df(settings=settings)
    if df.empty:
        return {"count": 0}

    out: Dict[str, Any] = {
        "count": int(len(df)),
        "avg_positive_pct": float(df["positive_pct"].mean()),
        "avg_negative_pct": float(df["negative_pct"].mean()),
        "avg_neutral_pct": float(df["neutral_pct"].mean()),
    }
    if "overall_label" in df.columns:
        out["label_counts"] = df["overall_label"].value_counts().to_dict()

    # Aspect mention rates from stored JSON
    aspect_mentions: Dict[str, int] = {}
    aspect_pos_sum: Dict[str, float] = {}
    aspect_n: Dict[str, int] = {}
    for _, row in df.iterrows():
        try:
            aspects = json.loads(row["aspects_json"]) if row["aspects_json"] else []
        except Exception:
            aspects = []
        for a in aspects:
            name = str(a.get("aspect", ""))
            if not name:
                continue
            if a.get("mentioned"):
                aspect_mentions[name] = aspect_mentions.get(name, 0) + 1
                if a.get("positive_pct") is not None:
                    aspect_pos_sum[name] = aspect_pos_sum.get(name, 0.0) + float(a["positive_pct"])
                    aspect_n[name] = aspect_n.get(name, 0) + 1
    out["aspect_mention_counts"] = aspect_mentions
    out["aspect_avg_positive_pct"] = {
        k: round(aspect_pos_sum[k] / aspect_n[k], 2) for k in aspect_n if aspect_n[k]
    }
    return out
