"""Aspect-based sentiment: detect product aspects and score localized text spans."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from app.sentiment_percentages import scores_to_percentages, scores_to_prob_triplet

# Aspect name -> surface cues (lowercase substrings). Extend for your domain.
ASPECT_LEXICON: Dict[str, List[str]] = {
    "quality & build": ["quality", "build", "material", "sturdy", "solid", "cheap feel", "premium"],
    "price & value": ["price", "value", "expensive", "cheap", "cost", "worth", "affordable", "overpriced"],
    "delivery & shipping": ["delivery", "shipping", "arrived", "package arrived", "courier", "on time"],
    "battery & power": ["battery", "charge", "charging", "power", "lasts", "drain", "usb-c", "adapter"],
    "screen & display": ["screen", "display", "brightness", "oled", "lcd", "resolution", "pixel"],
    "performance & speed": ["fast", "slow", "speed", "performance", "lag", "laggy", "snappy", "responsive"],
    "design & look": ["design", "look", "aesthetic", "color", "sleek", "bulky", "lightweight"],
    "customer service": ["support", "service", "customer", "refund", "warranty", "help desk", "rude"],
    "usability": ["easy", "intuitive", "confusing", "interface", "setup", "simple", "complicated"],
    "sound & audio": ["sound", "audio", "speaker", "volume", "bass", "noise", "microphone"],
    "software & bugs": ["software", "app", "bug", "update", "firmware", "crash", "glitch"],
    "packaging": ["packaging", "box", "damaged box", "wrapped"],
}


def split_sentences(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_matches(sentence: str, keywords: List[str]) -> bool:
    low = sentence.lower()
    return any(kw in low for kw in keywords)


def sentences_for_aspect(text: str, keywords: List[str]) -> List[str]:
    out: List[str] = []
    for sent in split_sentences(text):
        if _sentence_matches(sent, keywords):
            out.append(sent)
    if not out and any(kw in text.lower() for kw in keywords):
        # Fall back to full text if keywords appear but sentence split missed (e.g. no punctuation)
        return [text.strip()]
    return out


def aggregate_aspect_scores(
    sentences: List[str],
    predict_fn: Callable[[str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not sentences:
        return None
    pos_acc = neg_acc = neu_acc = 0.0
    n = 0
    for sent in sentences:
        res = predict_fn(sent)
        scores = res.get("scores") or {}
        pp, pn, pu = scores_to_prob_triplet(scores)
        pos_acc += pp
        neg_acc += pn
        neu_acc += pu
        n += 1
    if n == 0:
        return None
    pos_acc /= n
    neg_acc /= n
    neu_acc /= n
    merged = {"positive": pos_acc, "negative": neg_acc, "neutral": neu_acc}
    pct = scores_to_percentages(merged)
    dom = max(merged.items(), key=lambda kv: kv[1])[0]
    return {
        "dominant_label": dom,
        "positive_pct": pct["positive_pct"],
        "negative_pct": pct["negative_pct"],
        "neutral_pct": pct["neutral_pct"],
        "snippets": sentences[:3],
    }


def aspect_based_sentiment(
    text: str,
    predict_fn: Callable[[str], Dict[str, Any]],
    lexicon: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """Return one record per aspect in lexicon."""
    lex = lexicon or ASPECT_LEXICON
    rows: List[Dict[str, Any]] = []
    for aspect, keywords in lex.items():
        sents = sentences_for_aspect(text, keywords)
        if not sents:
            rows.append(
                {
                    "aspect": aspect,
                    "mentioned": False,
                    "dominant_label": None,
                    "positive_pct": None,
                    "negative_pct": None,
                    "neutral_pct": None,
                    "snippets": [],
                }
            )
            continue
        agg = aggregate_aspect_scores(sents, predict_fn)
        if not agg:
            rows.append(
                {
                    "aspect": aspect,
                    "mentioned": True,
                    "dominant_label": None,
                    "positive_pct": None,
                    "negative_pct": None,
                    "neutral_pct": None,
                    "snippets": sents[:3],
                }
            )
            continue
        rows.append(
            {
                "aspect": aspect,
                "mentioned": True,
                "dominant_label": agg["dominant_label"],
                "positive_pct": agg["positive_pct"],
                "negative_pct": agg["negative_pct"],
                "neutral_pct": agg["neutral_pct"],
                "snippets": agg["snippets"],
            }
        )
    return rows
