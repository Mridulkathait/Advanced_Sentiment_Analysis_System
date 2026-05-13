"""Load saved models and run inference for the Streamlit app."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from functools import cache

import joblib
import numpy as np

from preprocessing.pipeline import TextPreprocessor
from utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


@cache
def _bert_predictor():
    from bert.inference import BertSentimentPredictor

    return BertSentimentPredictor()


def list_ml_models(settings: Optional[Settings] = None) -> List[Path]:
    settings = settings or get_settings()
    d = settings.saved_models_dir / "ml"
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.joblib") if p.name != "best_metadata.json")


def load_ml_bundle(path: Path) -> Dict[str, Any]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    return {"pipeline": obj, "label_encoder": None}


def predict_ml(text: str, bundle: Dict[str, Any], preprocessor: TextPreprocessor) -> Dict[str, Any]:
    pipe = bundle["pipeline"]
    le: Optional[Any] = bundle.get("label_encoder")
    processed = preprocessor.transform(text)
    pred = pipe.predict([processed])[0]
    if le is not None:
        label = str(le.inverse_transform(np.asarray([pred]).astype(int))[0])
    else:
        label = str(pred)

    scores: Dict[str, float] = {}
    conf = 0.0
    clf = pipe.named_steps.get("clf", pipe)
    class_ids = getattr(clf, "classes_", None)

    if hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba([processed])[0]).ravel()
        if class_ids is None:
            class_ids = np.arange(len(proba))
        for cid, p in zip(class_ids, proba):
            name = str(le.inverse_transform(np.asarray([int(cid)]))[0]) if le is not None else str(cid)
            scores[name] = float(p)
        conf = float(np.max(proba))
    elif hasattr(pipe, "decision_function"):
        margins = np.asarray(pipe.decision_function([processed])[0]).ravel()
        ex = np.exp(margins - np.max(margins))
        proba = ex / ex.sum()
        if class_ids is None:
            class_ids = np.arange(len(proba))
        for cid, p in zip(class_ids, proba):
            name = str(le.inverse_transform(np.asarray([int(cid)]))[0]) if le is not None else str(cid)
            scores[name] = float(p)
        conf = float(np.max(proba))
    else:
        scores[label] = 1.0
        conf = 1.0

    return {"label": label, "confidence": conf, "scores": scores, "processed_preview": processed[:280]}


def predict_lstm(text: str, settings: Optional[Settings] = None) -> Optional[Dict[str, Any]]:
    settings = settings or get_settings()
    art_path = settings.saved_models_dir / "lstm" / "lstm_artifacts.joblib"
    model_path = settings.saved_models_dir / "lstm" / "lstm_best.keras"
    if not art_path.exists() or not model_path.exists():
        return None
    try:
        import tensorflow as tf
    except Exception:
        return None
    from deep_learning.lstm_model import prepare_sequences

    artifacts = joblib.load(art_path)
    tok = artifacts["tokenizer"]
    le = artifacts["label_encoder"]
    max_len = int(artifacts.get("max_len", settings.lstm_max_len))
    vocab_size = int(artifacts.get("vocab_size", settings.lstm_vocab_size))
    model = tf.keras.models.load_model(model_path)
    X, _ = prepare_sequences([text], tok, max_len, vocab_size, fit=False)
    proba = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(proba))
    label = str(le.inverse_transform([idx])[0])
    return {
        "label": label,
        "confidence": float(np.max(proba)),
        "scores": {str(c): float(p) for c, p in zip(le.classes_, proba)},
    }


def predict_bert(text: str) -> Optional[Dict[str, Any]]:
    try:
        from bert.inference import map_binary_to_three
    except Exception:
        return None
    try:
        pred = _bert_predictor().predict(text)
        lab = str(pred["label"])
        conf = float(pred["confidence"])
        scores = pred.get("scores") or {}
        if lab.lower() in {"positive", "negative", "neutral"}:
            return {"label": lab.lower(), "confidence": conf, "scores": scores, "raw": pred}
        label, conf2 = map_binary_to_three(lab, conf)
        merged_scores = scores if scores else {lab: conf}
        return {"label": label, "confidence": conf2, "scores": merged_scores, "raw": pred}
    except Exception as exc:
        logger.warning("BERT inference failed: %s", exc)
        return None


def load_leaderboard(settings: Optional[Settings] = None) -> List[dict]:
    settings = settings or get_settings()
    p = settings.reports_dir / "ml_leaderboard.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def keyword_highlights(processed: str, top_k: int = 8) -> List[str]:
    toks = [t for t in processed.split() if len(t) > 2]
    return toks[:top_k]


def compare_models(text: str, settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """Run every available backend on the same review for leaderboard-style comparison."""
    settings = settings or get_settings()
    pre = TextPreprocessor()
    rows: List[Dict[str, Any]] = []
    for path in list_ml_models(settings):
        bundle = load_ml_bundle(path)
        res = predict_ml(text, bundle, pre)
        rows.append(
            {
                "family": "Classical ML",
                "model": path.stem,
                "label": res.get("label"),
                "confidence": res.get("confidence"),
            }
        )
    lstm = predict_lstm(text, settings)
    if lstm:
        rows.append(
            {
                "family": "Deep Learning",
                "model": "BiLSTM",
                "label": lstm.get("label"),
                "confidence": lstm.get("confidence"),
            }
        )
    bert = predict_bert(text)
    if bert:
        rows.append(
            {
                "family": "Transformers",
                "model": "BERT / DistilBERT",
                "label": bert.get("label"),
                "confidence": bert.get("confidence"),
            }
        )
    return rows
