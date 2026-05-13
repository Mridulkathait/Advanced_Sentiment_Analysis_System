"""Generate exploratory charts and statistics into `reports/`."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing.dataset import load_reviews_csv
from preprocessing.pipeline import TextPreprocessor
from utils.config import get_settings

logger = logging.getLogger(__name__)


def _tokenize_for_freq(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", str(text).lower()) if len(t) > 2]


def _word_counts_for_subset(texts: List[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(_tokenize_for_freq(t))
    return c


def generate_eda(csv_path: Path) -> None:
    settings = get_settings()
    df = load_reviews_csv(csv_path)
    out_dir = settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Sentiment distribution
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=df, x="label", hue="label", palette="viridis", legend=False)
    plt.title("Sentiment distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_sentiment_dist.png", dpi=160)
    plt.close()

    try:
        import plotly.express as px

        fig_p = px.bar(
            df["label"].value_counts().reset_index(),
            x="label",
            y="count",
            color="label",
            title="Sentiment distribution (interactive)",
        )
        fig_p.write_html(out_dir / "eda_sentiment_dist.html", include_plotlyjs="cdn")
    except Exception as exc:
        logger.debug("Plotly sentiment chart skipped: %s", exc)

    # --- Review length
    df = df.copy()
    df["length"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()

    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x="label", y="length", hue="label", palette="mako", legend=False)
    plt.title("Review length (characters) by sentiment")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_length_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df, x="word_count", hue="label", bins=30, kde=True, element="step", legend=True)
    plt.title("Word count distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_word_count_hist.png", dpi=160)
    plt.close()

    # --- Processed corpus for aligned NLP stats
    pre = TextPreprocessor()
    df["processed"] = pre.transform_series(df["text"])

    overall = _word_counts_for_subset(df["processed"].tolist())
    pos_texts = df.loc[df["label"] == "positive", "processed"].tolist()
    neg_texts = df.loc[df["label"] == "negative", "processed"].tolist()
    neu_texts = df.loc[df["label"] == "neutral", "processed"].tolist()

    pos_c = _word_counts_for_subset(pos_texts)
    neg_c = _word_counts_for_subset(neg_texts)

    top_n = 20
    top_overall = overall.most_common(top_n)
    if top_overall:
        w, c = zip(*top_overall)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(c), y=list(w), color="#c0392b")
        plt.title(f"Top {top_n} words (processed corpus)")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / "eda_word_freq_top.png", dpi=160)
        plt.close()

    # Positive vs negative distinctive words (simple log-odds style ranking)
    def rel_freq(counter: Counter, denom: int) -> Dict[str, float]:
        return {w: counter[w] / max(denom, 1) for w in counter}

    pos_denom = sum(pos_c.values()) or 1
    neg_denom = sum(neg_c.values()) or 1
    pos_rf = rel_freq(pos_c, pos_denom)
    neg_rf = rel_freq(neg_c, neg_denom)
    vocab = set(pos_c.keys()) | set(neg_c.keys())
    scored: List[Tuple[str, float]] = []
    for w in vocab:
        scored.append((w, pos_rf.get(w, 0) - neg_rf.get(w, 0)))
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    pos_skew = [w for w, s in scored if s > 0][:15]
    neg_skew = [w for w, s in scored if s < 0][:15]

    if pos_skew:
        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=[pos_c[w] for w in pos_skew],
            y=pos_skew,
            color="#2ecc71",
        )
        plt.title("Frequent words skewing positive (processed)")
        plt.tight_layout()
        plt.savefig(out_dir / "eda_words_positive_skew.png", dpi=160)
        plt.close()

    if neg_skew:
        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=[neg_c[w] for w in neg_skew],
            y=neg_skew,
            color="#e74c3c",
        )
        plt.title("Frequent words skewing negative (processed)")
        plt.tight_layout()
        plt.savefig(out_dir / "eda_words_negative_skew.png", dpi=160)
        plt.close()

    # --- Word clouds
    try:
        from wordcloud import WordCloud

        def save_wc(text_blob: str, fname: str, title: str) -> None:
            if not text_blob.strip():
                return
            wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(
                text_blob
            )
            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=140)
            plt.close()

        save_wc(" ".join(df["processed"].tolist()), "eda_wordcloud_all.png", "Word cloud — full corpus")
        if pos_texts:
            save_wc(" ".join(pos_texts), "eda_wordcloud_positive.png", "Word cloud — positive")
        if neg_texts:
            save_wc(" ".join(neg_texts), "eda_wordcloud_negative.png", "Word cloud — negative")
        if neu_texts:
            save_wc(" ".join(neu_texts), "eda_wordcloud_neutral.png", "Word cloud — neutral")
    except Exception as exc:
        logger.warning("Word cloud generation failed: %s", exc)

    stats = {
        "rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "mean_chars": float(df["length"].mean()),
        "median_chars": float(df["length"].median()),
        "mean_words": float(df["word_count"].mean()),
        "median_words": float(df["word_count"].median()),
        "top_words_overall": [{"token": w, "count": int(c)} for w, c in overall.most_common(50)],
        "positive_skew_tokens": pos_skew[:30],
        "negative_skew_tokens": neg_skew[:30],
    }
    (out_dir / "eda_dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("EDA artifacts written to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_eda(Path("data/sample_reviews.csv"))
