# Advanced Sentiment Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/sklearn-ML-orange)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/HF-Transformers-yellow)](https://huggingface.co/docs/transformers)

End-to-end sentiment intelligence for product reviews: classical ML with tuning, deep BiLSTM, transformer inference and fine-tuning, analytics dashboards, optional LIME explanations, and a polished Streamlit client.

## Highlights

- **Preprocessing:** HTML/URL cleanup, emoji removal, normalization, tokenization, stopwords, stemming/lemmatization, dedupe, balancing, dynamic CSV loading.
- **EDA:** rich CLI (`python main.py eda`) with distributions, lengths, word frequencies, positive vs negative skew charts, word clouds, Plotly HTML export, and `reports/eda_dataset_stats.json`.
- **Features:** TF-IDF / BoW with uni-, bi-, and tri-grams (`preprocessing/feature_engineering.py`).
- **Models:** Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost, BiLSTM (TensorFlow), DistilBERT (HuggingFace pipeline + optional local fine-tune).
- **Evaluation:** Stratified K-Fold, `GridSearchCV`, confusion matrix for the best ML bundle, macro precision/recall/F1 in the leaderboard, downloadable classification JSON.
- **App:** single-review and **Model arena** comparison, **positive / negative / neutral %**, **aspect-based sentiment** (lexicon-grounded per-aspect scores on matching sentences), **SQLite review history** (`data/review_history.sqlite3`) with **Insights & history** analytics, bulk CSV, JSON export, light/dark theme, Plotly dashboards.

## Architecture

```
data/               # CSV datasets (review + sentiment columns)
notebooks/          # EDA starter notebook
scripts/            # CLI helpers (EDA orchestration)
preprocessing/      # Cleaning pipeline + vectorizer builders + dataset IO
models/             # Architecture anchor / documentation package
training/           # ML training, tuning, and serialization
evaluation/         # Metrics + plotting helpers
deep_learning/      # BiLSTM training + sequence utilities
bert/               # Transformer inference + HuggingFace Trainer fine-tune
app/                # Streamlit UI + model service
utils/              # Config, logging, caching, optional explainability
assets/             # Screenshots / branding (populate locally)
saved_models/       # Serialized weights (generated)
reports/            # Figures, leaderboards, EDA stats (generated)
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py eda
python main.py train-ml --csv data/sample_reviews.csv --confusion
# Optional GPU-friendly fine-tune (CPU works, slower):
python main.py train-bert --csv data/sample_reviews.csv --epochs 2 --batch-size 4
streamlit run app/streamlit_app.py
```

> **PyTorch:** install the wheel that matches your platform from [pytorch.org](https://pytorch.org) if `pip install torch` fails.

## CLI

| Command | Purpose |
| --- | --- |
| `python main.py eda` | Full EDA pack into `reports/` (PNG/HTML/JSON) |
| `python main.py train-ml [--balance] [--confusion]` | Train/compare ML models → `reports/ml_leaderboard.json`, `reports/ml_best_classification.json`, `reports/confusion_best_ml.png` |
| `python main.py train-lstm [--balance]` | Train BiLSTM (TensorFlow) → `saved_models/lstm/` |
| `python main.py train-bert [--epochs N] [--batch-size B] [--max-samples M]` | Fine-tune DistilBERT on your CSV → `saved_models/bert/finetuned/` (overrides default HF head in the app) |

## Web App

- **Analyze → Single review:** live prediction, class scores, keyword highlights, optional **LIME** explanations for ML bundles, downloadable JSON report.
- **Analyze → Model arena:** run every available ML/LSTM/BERT backend on the same text with a comparison chart.
- **Dashboards:** macro F1 / precision / recall bars, confusion matrix image, classification report JSON, dataset pie + word cloud, session sentiment trend (from in-app history).
- **Bulk CSV:** score a `review` column and download `predictions.csv`.
- **About:** stack overview and recommended workflow.

## Screenshots

Add your own captures to `assets/` after your first run:

```markdown
![Analyze](assets/analyze.png)
![Dashboard](assets/dashboard.png)
```

## Model comparison & artifacts

- Leaderboard: `reports/ml_leaderboard.json` (includes `test_precision_macro` and `test_recall_macro` for each trained bundle).
- Best bundle metadata: `saved_models/ml/best_metadata.json`.
- Best test-set report: `reports/ml_best_classification.json`.
- EDA statistics: `reports/eda_dataset_stats.json`.

## Explainability

- Keyword highlights are always available (post-processed tokens).
- Optional **LIME** for classical ML pipelines (`pip install lime`, enabled via checkbox in the app).
- `shap` is commented in `requirements.txt` for heavier setups.

## Notes

- If `saved_models/bert/finetuned/` contains a `config.json`, the Streamlit app prefers that checkpoint; otherwise it falls back to `distilbert-base-uncased-finetuned-sst-2-english` with a neutral mapping heuristic.
- LSTM consumes raw `text` while classical models use the `processed` column to stay aligned with the Streamlit preprocessor.

## Future scope

- REST API (FastAPI) + Docker images for deployment.
- SHAP integration for sparse linear models on managed baselines.
- Automated tests and CI workflows.

## License

MIT — add a `LICENSE` file if you plan to distribute the project publicly.
