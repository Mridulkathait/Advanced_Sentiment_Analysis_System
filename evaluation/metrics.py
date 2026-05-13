"""Evaluation utilities: confusion matrix, plots, metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
    out_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", out_path)
    return fig


def plot_training_history(history: dict, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if "loss" not in history:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["loss"], label="train")
    if "val_loss" in history:
        ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    if "accuracy" in history:
        ax[1].plot(history["accuracy"], label="train")
        if "val_accuracy" in history:
            ax[1].plot(history["val_accuracy"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
    plt.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)
