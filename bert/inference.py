"""HuggingFace transformer inference and optional fine-tuning helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import pipeline

    HAS_HF = True
except Exception:
    HAS_HF = False


def resolve_bert_model_path(model_name: Optional[str] = None, device: Optional[int] = None) -> Tuple[str, Optional[int]]:
    """Prefer locally fine-tuned weights when present; otherwise fall back to configured HF model."""
    from utils.config import get_settings

    settings = get_settings()
    local_dir = settings.saved_models_dir / "bert" / "finetuned"
    if (local_dir / "config.json").exists():
        logger.info("Using fine-tuned BERT at %s", local_dir)
        return str(local_dir), device
    resolved = model_name or settings.bert_model_name
    return resolved, device


class BertSentimentPredictor:
    """Loads a sentiment model (local fine-tune preferred) for inference."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None) -> None:
        if not HAS_HF:
            raise RuntimeError("transformers/torch not installed")
        resolved, dev = resolve_bert_model_path(model_name, device)
        self.model_name = resolved
        self.device = 0 if dev is None and torch.cuda.is_available() else dev
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device if self.device is not None else -1,
            truncation=True,
        )

    def predict(self, text: str) -> Dict[str, Any]:
        raw = self._pipe(text[:2000], top_k=None)
        scores_block = raw[0] if isinstance(raw, list) else raw
        if isinstance(scores_block, dict):
            scores_block = [scores_block]
        best = max(scores_block, key=lambda x: float(x.get("score", 0.0)))
        label = str(best.get("label", "UNKNOWN"))
        score = float(best.get("score", 0.0))
        lab_lower = label.lower()
        if lab_lower in {"positive", "negative", "neutral"}:
            dist = {str(s["label"]).lower(): float(s["score"]) for s in scores_block}
            return {"label": lab_lower, "confidence": score, "scores": dist, "raw": raw}
        dist = {str(s["label"]): float(s["score"]) for s in scores_block}
        return {"label": label, "confidence": score, "scores": dist, "raw": raw}


def map_binary_to_three(label: str, confidence: float) -> Tuple[str, float]:
    """Map SST-2 style POSITIVE/NEGATIVE to project schema with neutral band."""
    lab = str(label).upper()
    if lab == "POSITIVE":
        return "positive", confidence
    if lab == "NEGATIVE":
        return "negative", confidence
    if confidence >= 0.9:
        return lab.lower(), confidence
    return "neutral", 1.0 - abs(confidence - 0.5) * 2


def fine_tune_stub() -> None:
    """Backward-compatible hook; prefer `bert.fine_tune.run_bert_finetune`."""
    logger.info("Use `python main.py train-bert` for HuggingFace Trainer fine-tuning.")
