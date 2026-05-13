"""Fine-tune DistilBERT (or compatible encoder) for 3-class review sentiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from preprocessing.dataset import load_reviews_csv
from utils.config import Settings, get_settings

logger = logging.getLogger(__name__)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def run_bert_finetune(
    csv_path: Path,
    epochs: int = 3,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    learning_rate: float = 2e-5,
    model_name: str = "distilbert-base-uncased",
    settings: Optional[Settings] = None,
) -> Path:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install transformers, torch, and datasets for BERT fine-tuning.") from exc

    settings = settings or get_settings()
    out_dir = settings.saved_models_dir / "bert" / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviews_csv(csv_path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=settings.random_state).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    labels = [LABEL2ID[str(l)] for l in df["label"]]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def build_hf_dataset(tx: list[str], ly: list[int]) -> Dataset:
        ds = Dataset.from_dict({"text": tx, "labels": ly})

        def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
            enc = tokenizer(
                batch["text"],
                truncation=True,
                padding=False,
                max_length=settings.bert_max_length,
            )
            enc["labels"] = batch["labels"]
            return enc

        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds = build_hf_dataset(train_texts, train_labels)
    val_ds = build_hf_dataset(val_texts, val_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels_arr, preds)),
            "f1_macro": float(f1_score(labels_arr, preds, average="macro", zero_division=0)),
        }

    use_fp16 = bool(torch.cuda.is_available())
    base_args = dict(
        output_dir=str(out_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=float(epochs),
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=max(10, len(train_ds) // (batch_size * 5)),
        save_total_limit=2,
        fp16=use_fp16,
        report_to=[],
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **base_args)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **base_args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    mapping_path = out_dir / "label_mapping.json"
    mapping_path.write_text(
        json.dumps({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}}, indent=2),
        encoding="utf-8",
    )
    logger.info("Fine-tuned model saved to %s", out_dir)
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bert_finetune(Path("data/sample_reviews.csv"), epochs=2, batch_size=4)
