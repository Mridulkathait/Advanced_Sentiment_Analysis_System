"""Optional local explanations (LIME) for classical ML pipelines."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from preprocessing.pipeline import TextPreprocessor

logger = logging.getLogger(__name__)


def explain_with_lime(
    text: str,
    pipeline: Any,
    preprocessor: TextPreprocessor,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    num_samples: int = 5000,
) -> Optional[Dict[str, Any]]:
    """Return LIME weights for processed tokens; None if lime is unavailable."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.info("lime not installed; skipping local explanation.")
        return None

    processed = preprocessor.transform(text)
    if not processed.strip():
        return None

    if class_names is None:
        clf = pipeline.named_steps.get("clf", pipeline)
        classes = getattr(clf, "classes_", None)
        if classes is not None:
            class_names = [str(c) for c in classes.tolist()]
        else:
            class_names = ["negative", "neutral", "positive"]

    explainer = LimeTextExplainer(class_names=class_names)

    def classifier_fn(texts: List[str]) -> Any:
        proc = [preprocessor.transform(t) for t in texts]
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(proc)
        margins = pipeline.decision_function(proc)
        import numpy as np

        ex = np.exp(margins - np.max(margins, axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    exp = explainer.explain_instance(
        text,
        classifier_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=None,
    )
    label_id = exp.top_label
    weights = exp.as_list(label=label_id)
    return {
        "label_id": int(label_id),
        "label_name": class_names[int(label_id)] if label_id < len(class_names) else str(label_id),
        "weights": weights,
    }
