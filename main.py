"""CLI entrypoint for training, evaluation exports, and EDA."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import plot_confusion_matrix, plot_training_history
from preprocessing.dataset import build_splits, load_reviews_csv, preprocess_dataframe
from preprocessing.pipeline import TextPreprocessor
from training.ml_trainer import run_ml_benchmark
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger("main")


def cmd_train_ml(args: argparse.Namespace) -> None:
    settings = get_settings()
    csv_path = Path(args.csv)
    df = load_reviews_csv(csv_path)
    if args.balance:
        from preprocessing.dataset import balance_classes

        df = balance_classes(df)
    pre = TextPreprocessor()
    df = preprocess_dataframe(df, pre)
    train_df, test_df = build_splits(df, test_size=settings.test_size, random_state=settings.random_state)
    results, best = run_ml_benchmark(train_df, test_df)
    logger.info("Trained %s ML configurations. Best bundle ready for inference.", len(results))
    if best and args.confusion:
        import numpy as np

        pipe = best["pipeline"]
        le = best["label_encoder"]
        y_pred = pipe.predict(test_df["processed"])
        if le is not None:
            y_pred = le.inverse_transform(np.asarray(y_pred).astype(int))
        labels = sorted(test_df["label"].unique().tolist())
        plot_confusion_matrix(
            test_df["label"],
            y_pred,
            labels=labels,
            out_path=settings.reports_dir / "confusion_best_ml.png",
            title="Best ML model",
        )


def cmd_train_lstm(args: argparse.Namespace) -> None:
    settings = get_settings()
    from preprocessing.dataset import load_reviews_csv, preprocess_dataframe
    from deep_learning.lstm_model import train_lstm

    df = load_reviews_csv(Path(args.csv))
    if args.balance:
        from preprocessing.dataset import balance_classes

        df = balance_classes(df)
    pre = TextPreprocessor(use_lemmatization=False, use_stemming=False, remove_stopwords=False)
    df = preprocess_dataframe(df, pre)
    train_df, test_df = build_splits(df, test_size=0.2, random_state=settings.random_state)
    model, history = train_lstm(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_df["text"].tolist(),
        test_df["label"].tolist(),
    )
    plot_training_history(history, settings.reports_dir / "lstm_history")
    logger.info("LSTM training complete.")


def cmd_eda(args: argparse.Namespace) -> None:
    from scripts.run_eda import generate_eda

    generate_eda(Path(args.csv))


def cmd_train_bert(args: argparse.Namespace) -> None:
    from bert.fine_tune import run_bert_finetune

    out = run_bert_finetune(
        Path(args.csv),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_samples=args.max_samples,
        learning_rate=float(args.lr),
        model_name=str(args.model),
    )
    logger.info("BERT fine-tune artifacts at %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced Sentiment Analysis System")
    sub = parser.add_subparsers(dest="command")

    p_ml = sub.add_parser("train-ml", help="Train & compare ML models")
    p_ml.add_argument("--csv", type=str, default=str(ROOT / "data" / "sample_reviews.csv"))
    p_ml.add_argument("--balance", action="store_true", help="Oversample minority classes")
    p_ml.add_argument("--confusion", action="store_true", help="Save confusion matrix for best model")

    p_lstm = sub.add_parser("train-lstm", help="Train bidirectional LSTM")
    p_lstm.add_argument("--csv", type=str, default=str(ROOT / "data" / "sample_reviews.csv"))
    p_lstm.add_argument("--balance", action="store_true")

    p_eda = sub.add_parser("eda", help="Generate EDA plots into reports/")
    p_eda.add_argument("--csv", type=str, default=str(ROOT / "data" / "sample_reviews.csv"))

    p_bert = sub.add_parser("train-bert", help="Fine-tune DistilBERT on your CSV (3-class)")
    p_bert.add_argument("--csv", type=str, default=str(ROOT / "data" / "sample_reviews.csv"))
    p_bert.add_argument("--epochs", type=int, default=3)
    p_bert.add_argument("--batch-size", type=int, default=8)
    p_bert.add_argument("--lr", type=float, default=2e-5)
    p_bert.add_argument("--max-samples", type=int, default=None, help="Optional cap for faster experiments")
    p_bert.add_argument("--model", type=str, default="distilbert-base-uncased", help="HF encoder checkpoint")

    args = parser.parse_args()
    if args.command == "train-ml":
        cmd_train_ml(args)
    elif args.command == "train-lstm":
        cmd_train_lstm(args)
    elif args.command == "eda":
        cmd_eda(args)
    elif args.command == "train-bert":
        cmd_train_bert(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
