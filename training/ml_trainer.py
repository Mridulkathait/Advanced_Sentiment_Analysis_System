"""Train and compare classical ML models with tuning and CV."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from preprocessing.feature_engineering import build_vectorizer
from utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


@dataclass
class ModelResult:
    name: str
    vectorizer: str
    best_params: Dict[str, Any]
    cv_mean_f1: float
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    model_path: str


def _build_estimator(name: str) -> Any:
    name = name.lower()
    if name == "naive_bayes":
        return MultinomialNB()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    if name == "svm":
        return LinearSVC(class_weight="balanced", random_state=42, max_iter=4000, dual=False)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unknown model: {name}")


def _param_grid(name: str) -> Dict[str, List[Any]]:
    name = name.lower()
    if name == "naive_bayes":
        return {"clf__alpha": [0.1, 0.5, 1.0]}
    if name == "logistic_regression":
        return {"clf__C": [0.25, 1.0, 4.0]}
    if name == "svm":
        return {"clf__C": [0.25, 1.0, 4.0]}
    if name == "random_forest":
        return {"clf__max_depth": [None, 16, 24], "clf__min_samples_leaf": [1, 2]}
    if name == "xgboost":
        return {"clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]}
    return {}


def train_model_with_grid(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    vec_name: str,
    vectorizer,
    settings: Settings,
    label_encoder: LabelEncoder,
) -> Tuple[Pipeline, ModelResult]:
    clf = _build_estimator(model_name)
    pipe = Pipeline([("vec", vectorizer), ("clf", clf)])
    grid = _param_grid(model_name)
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    if grid:
        search = GridSearchCV(
            pipe,
            grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = dict(search.best_params_)
        cv_mean = float(search.best_score_)
    else:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_mean = float(np.mean(scores))
        pipe.fit(X_train, y_train)
        best = pipe
        best_params = {}

    y_pred = best.predict(X_test)
    y_true_labels = label_encoder.inverse_transform(np.asarray(y_test))
    y_pred_labels = label_encoder.inverse_transform(np.asarray(y_pred))
    acc = float(accuracy_score(y_true_labels, y_pred_labels))
    f1m = float(f1_score(y_true_labels, y_pred_labels, average="macro"))
    prec = float(precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    rec = float(recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0))
    out_dir = settings.saved_models_dir / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{model_name}_{vec_name}.joblib"
    path = out_dir / fname
    joblib.dump({"pipeline": best, "label_encoder": label_encoder}, path)
    result = ModelResult(
        name=model_name,
        vectorizer=vec_name,
        best_params=best_params,
        cv_mean_f1=cv_mean,
        test_accuracy=acc,
        test_precision_macro=prec,
        test_recall_macro=rec,
        test_f1_macro=f1m,
        model_path=str(path),
    )
    logger.info(
        "%s (%s) | CV F1 macro=%.4f | test acc=%.4f | P/R(macro)=%.4f/%.4f",
        model_name,
        vec_name,
        cv_mean,
        acc,
        prec,
        rec,
    )
    return best, result


def run_ml_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "processed",
    label_col: str = "label",
    settings: Optional[Settings] = None,
) -> Tuple[List[ModelResult], Optional[dict]]:
    settings = settings or get_settings()
    X_train = train_df[text_col].astype(str)
    y_train = train_df[label_col].astype(str)
    X_test = test_df[text_col].astype(str)
    y_test = test_df[label_col].astype(str)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    vectorizers = {
        "tfidf_12": build_vectorizer("tfidf", ngram_range=(1, 2), max_features=settings.max_features_tfidf),
        "tfidf_13": build_vectorizer("tfidf", ngram_range=(1, 3), max_features=settings.max_features_tfidf),
        "bow_12": build_vectorizer("bow", ngram_range=(1, 2), max_features=min(15000, settings.max_features_tfidf)),
    }
    model_names = ["naive_bayes", "logistic_regression", "svm", "random_forest"]
    if HAS_XGB:
        model_names.append("xgboost")

    results: List[ModelResult] = []
    best_tuple: Optional[Tuple[float, dict, ModelResult]] = None

    for vec_name, vec in vectorizers.items():
        for m in model_names:
            try:
                vec_fresh = vec.__class__(**vec.get_params())
                best_pipe, res = train_model_with_grid(
                    X_train,
                    y_train_enc,
                    X_test,
                    y_test_enc,
                    m,
                    vec_name,
                    vec_fresh,
                    settings,
                    label_encoder,
                )
                results.append(res)
                key = res.test_f1_macro
                bundle = {"pipeline": best_pipe, "label_encoder": label_encoder}
                if best_tuple is None or key > best_tuple[0]:
                    best_tuple = (key, bundle, res)
            except Exception as exc:
                logger.exception("Failed training %s with %s: %s", m, vec_name, exc)

    leaderboard_path = settings.reports_dir / "ml_leaderboard.json"
    leaderboard_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    best_model = best_tuple[1] if best_tuple else None
    if best_tuple:
        best_pipe = best_tuple[1]["pipeline"]
        le_enc = best_tuple[1]["label_encoder"]
        y_pred = best_pipe.predict(X_test)
        y_true_labels = le_enc.inverse_transform(np.asarray(y_test))
        y_pred_labels = le_enc.inverse_transform(np.asarray(y_pred))
        labels = sorted(set(y_true_labels.tolist()) | set(y_pred_labels.tolist()))
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
        (settings.reports_dir / "ml_best_classification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        try:
            from evaluation.metrics import plot_confusion_matrix

            plot_confusion_matrix(
                y_true_labels,
                y_pred_labels,
                labels=labels,
                out_path=settings.reports_dir / "confusion_best_ml.png",
                title="Best ML model (test set)",
            )
        except Exception as exc:
            logger.warning("Could not render best-model confusion matrix: %s", exc)

        meta = {
            "best_model_name": best_tuple[2].name,
            "best_vectorizer": best_tuple[2].vectorizer,
            "best_params": best_tuple[2].best_params,
            "path": best_tuple[2].model_path,
            "test_accuracy": best_tuple[2].test_accuracy,
            "test_f1_macro": best_tuple[2].test_f1_macro,
            "test_precision_macro": best_tuple[2].test_precision_macro,
            "test_recall_macro": best_tuple[2].test_recall_macro,
        }
        (settings.saved_models_dir / "ml" / "best_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return results, best_model


def classification_report_dict(y_true, y_pred) -> Dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
