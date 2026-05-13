"""Map classifier score dicts to positive / negative / neutral percentages (0–100)."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(k).lower())


def scores_to_prob_triplet(scores: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return (p_pos, p_neg, p_neu) that sum to 1.0 when possible."""
    pos = neg = neu = 0.0
    if not scores:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    for raw_k, raw_v in scores.items():
        try:
            v = float(raw_v)
        except (TypeError, ValueError):
            continue
        k = _norm_key(raw_k)
        if k in {"positive", "pos", "label2", "stars5", "star5", "label1"} or (
            "positive" in k and "negative" not in k and "neutral" not in k
        ):
            pos += v
        elif k in {"negative", "neg", "label0", "stars1", "star1"} or (
            "negative" in k and "neutral" not in k
        ):
            neg += v
        elif "neutral" in k or "neu" in k or k in {"stars3"}:
            neu += v

    total = pos + neg + neu
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return pos / total, neg / total, neu / total


def scores_to_percentages(scores: Dict[str, Any]) -> Dict[str, float]:
    """Human-readable percentages summing to ~100."""
    p_pos, p_neg, p_neu = scores_to_prob_triplet(scores)
    return {
        "positive_pct": round(100.0 * p_pos, 2),
        "negative_pct": round(100.0 * p_neg, 2),
        "neutral_pct": round(100.0 * p_neu, 2),
    }
