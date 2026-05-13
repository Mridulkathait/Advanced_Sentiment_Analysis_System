"""Bidirectional LSTM classifier for text sentiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    HAS_TF = True
except Exception:
    HAS_TF = False


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    num_classes: int,
    lstm_units: int = 64,
    dropout: float = 0.35,
) -> Any:
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed")
    inp = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_sequences(
    texts: List[str],
    tokenizer: Optional[Any],
    max_len: int,
    vocab_size: int,
    fit: bool = False,
) -> Tuple[np.ndarray, Any]:
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed")
    tok = tokenizer or Tokenizer(num_words=vocab_size, oov_token="<unk>")
    if fit:
        tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded, tok


def train_lstm(
    train_texts: List[str],
    y_train,
    val_texts: List[str],
    y_val,
    settings=None,
    out_dir: Optional[Path] = None,
):
    if not HAS_TF:
        raise RuntimeError("TensorFlow is not installed")
    from utils.config import get_settings

    settings = settings or get_settings()
    out_dir = out_dir or (settings.saved_models_dir / "lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    X_train, tok = prepare_sequences(
        train_texts, None, settings.lstm_max_len, settings.lstm_vocab_size, fit=True
    )
    X_val, _ = prepare_sequences(val_texts, tok, settings.lstm_max_len, settings.lstm_vocab_size, fit=False)

    model = build_lstm_model(
        vocab_size=settings.lstm_vocab_size,
        embedding_dim=settings.lstm_embedding_dim,
        max_len=settings.lstm_max_len,
        num_classes=len(le.classes_),
        lstm_units=settings.lstm_units,
    )

    ckpt = ModelCheckpoint(filepath=str(out_dir / "lstm_best.keras"), save_best_only=True, monitor="val_loss")
    es = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")

    history = model.fit(
        X_train,
        y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=settings.lstm_epochs,
        batch_size=settings.lstm_batch_size,
        callbacks=[ckpt, es],
        verbose=1,
    )
    model.save(out_dir / "lstm_final.keras")
    import joblib

    joblib.dump(
        {
            "tokenizer": tok,
            "label_encoder": le,
            "max_len": settings.lstm_max_len,
            "vocab_size": settings.lstm_vocab_size,
        },
        out_dir / "lstm_artifacts.joblib",
    )
    return model, history.history
