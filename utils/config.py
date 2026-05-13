"""Central configuration with sensible defaults for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Application and training settings."""

    project_root: Path = field(default_factory=_project_root)
    data_dir: Path = field(init=False)
    saved_models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)

    text_column: str = "review"
    label_column: str = "sentiment"
    positive_labels: List[str] = field(default_factory=lambda: ["positive", "pos", "1", "Positive"])
    negative_labels: List[str] = field(default_factory=lambda: ["negative", "neg", "-1", "Negative"])
    neutral_labels: List[str] = field(default_factory=lambda: ["neutral", "neu", "0", "Neutral"])

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_features_tfidf: int = 20000
    lstm_max_len: int = 200
    lstm_vocab_size: int = 20000
    lstm_embedding_dim: int = 128
    lstm_units: int = 64
    lstm_epochs: int = 15
    lstm_batch_size: int = 32

    bert_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    bert_max_length: int = 128

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.saved_models_dir = self.project_root / "saved_models"
        self.reports_dir = self.project_root / "reports"
        self.assets_dir = self.project_root / "assets"
        for p in (self.data_dir, self.saved_models_dir, self.reports_dir, self.assets_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
