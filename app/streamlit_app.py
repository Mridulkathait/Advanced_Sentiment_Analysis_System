"""Streamlit front-end for the Advanced Sentiment Analysis System."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on path when launched via `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from app.aspect_analysis import aspect_based_sentiment
from app.review_store import aggregate_insights, clear_all, init_db, load_reviews_df, save_review
from app.sentiment_percentages import scores_to_percentages
from app.model_service import (
    compare_models,
    keyword_highlights,
    list_ml_models,
    load_leaderboard,
    load_ml_bundle,
    predict_bert,
    predict_lstm,
    predict_ml,
)
from preprocessing.pipeline import TextPreprocessor
from utils.config import get_settings


def inject_css(theme: str) -> None:
    is_dark = theme == "dark"
    bg = "#0b1220" if is_dark else "#f4f6fb"
    card = "#121a2b" if is_dark else "#ffffff"
    text = "#e8eefc" if is_dark else "#1b1f2a"
    accent = "#6c5ce7"
    accent2 = "#00cec9"
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --card: {card};
            --text: {text};
            --accent: {accent};
            --accent2: {accent2};
        }}
        .stApp {{
            background: radial-gradient(1200px 600px at 10% -10%, rgba(108,92,231,0.25), transparent),
                        radial-gradient(900px 500px at 90% 0%, rgba(0,206,201,0.18), transparent),
                        var(--bg);
            color: var(--text);
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--accent2);
        }}
        .hero {{
            padding: 1.4rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(120deg, rgba(108,92,231,0.35), rgba(0,206,201,0.25));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }}
        .card {{
            background: var(--card);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 10px 40px rgba(0,0,0,0.18);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_preprocessor() -> TextPreprocessor:
    return TextPreprocessor()


@st.cache_resource
def load_cached_ml(path_str: str):
    return load_ml_bundle(Path(path_str))


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"


def add_history(record: dict) -> None:
    st.session_state.history.insert(0, record)
    st.session_state.history = st.session_state.history[:50]


def run_prediction(text: str, model_choice: str, ml_path: str | None) -> dict:
    pre = get_preprocessor()
    if model_choice.startswith("ML |"):
        if not ml_path:
            return {"label": "n/a", "confidence": 0.0, "scores": {}}
        bundle = load_cached_ml(ml_path)
        return predict_ml(text, bundle, pre)
    if model_choice == "LSTM (trained)":
        return predict_lstm(text) or {"label": "n/a", "confidence": 0.0, "scores": {}}
    if model_choice == "BERT / Transformer":
        return predict_bert(text) or {"label": "n/a", "confidence": 0.0, "scores": {}}
    return {"label": "unknown", "confidence": 0.0, "scores": {}}


def build_prediction_report(
    text: str,
    model_choice: str,
    result: dict,
    percentages: dict,
    aspects: list,
) -> dict:
    return {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "model": model_choice,
        "input_excerpt": text[:2000],
        "overall_percentages": percentages,
        "aspect_based_sentiment": aspects,
        "prediction": {
            "label": result.get("label"),
            "confidence": result.get("confidence"),
            "scores": result.get("scores"),
        },
    }


def main() -> None:
    st.set_page_config(
        page_title="Advanced Sentiment Analysis",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    inject_css(st.session_state.theme)
    settings = get_settings()
    init_db(settings)

    st.sidebar.title("Navigation")
    st.session_state.theme = st.sidebar.radio(
        "Theme", ["dark", "light"], index=0 if st.session_state.theme == "dark" else 1
    )
    inject_css(st.session_state.theme)
    page = st.sidebar.radio(
        "Pages", ["Analyze", "Dashboards", "Insights & history", "Bulk CSV", "About"]
    )

    ml_paths = list_ml_models(settings)
    ml_labels = [f"ML | {p.stem}" for p in ml_paths]
    model_options = ml_labels + ["LSTM (trained)", "BERT / Transformer"]
    if not ml_labels:
        st.sidebar.info("No trained ML bundles found yet. Run `python main.py train-ml`.")

    st.markdown(
        '<div class="hero"><h1 style="margin:0;">Advanced Sentiment Analysis System</h1>'
        "<p style='opacity:.85;margin:.4rem 0 0;'>Overall + aspect sentiment, percentage breakdowns, persistent history, and analytics.</p></div>",
        unsafe_allow_html=True,
    )

    if page == "About":
        st.markdown(
            """
            **Stack:** modular preprocessing, classical ML with tuning, deep BiLSTM, HuggingFace transformers,
            Plotly dashboards, optional LIME explanations, and exportable prediction reports.

            **Workflow:** run `python main.py eda`, `python main.py train-ml`, optionally `python main.py train-bert`,
            then launch this app. Fine-tuned weights in `saved_models/bert/finetuned/` override the default SST-2 head.

            **This session:** overall sentiment is shown as **positive / negative / neutral percentages**. **Aspect-based**
            sentiment scores each product aspect (battery, price, etc.) on sentences that mention it. All analyzed
            reviews are stored in `data/review_history.sqlite3` for the **Insights & history** page.
            """
        )
        return

    if page == "Dashboards":
        leaderboard = load_leaderboard(settings)
        if leaderboard:
            df = pd.DataFrame(leaderboard)
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = px.bar(df, x="name", y="test_f1_macro", color="vectorizer", title="Model F1 (macro)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "test_precision_macro" in df.columns:
                    figp = px.bar(df, x="name", y="test_precision_macro", color="vectorizer", title="Precision (macro)")
                    st.plotly_chart(figp, use_container_width=True)
                else:
                    fig2 = px.bar(df, x="name", y="test_accuracy", color="vectorizer", title="Accuracy")
                    st.plotly_chart(fig2, use_container_width=True)
            with c3:
                if "test_recall_macro" in df.columns:
                    figr = px.bar(df, x="name", y="test_recall_macro", color="vectorizer", title="Recall (macro)")
                    st.plotly_chart(figr, use_container_width=True)
                else:
                    fig2 = px.bar(df, x="name", y="test_accuracy", color="vectorizer", title="Accuracy")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train ML models to populate comparison charts (`python main.py train-ml`).")

        cm_path = settings.reports_dir / "confusion_best_ml.png"
        if cm_path.exists():
            st.subheader("Confusion matrix (best ML)")
            st.image(str(cm_path), use_container_width=True)

        cls_path = settings.reports_dir / "ml_best_classification.json"
        if cls_path.exists():
            st.subheader("Best model — classification report (test)")
            st.json(json.loads(cls_path.read_text(encoding="utf-8")))

        sample_path = settings.data_dir / "sample_reviews.csv"
        if sample_path.exists():
            df_raw = pd.read_csv(sample_path)
            if "sentiment" in df_raw.columns:
                figp = px.pie(df_raw, names="sentiment", title="Dataset sentiment mix (sample)")
                st.plotly_chart(figp, use_container_width=True)
                try:
                    wc = WordCloud(width=1000, height=500, background_color=None, mode="RGBA").generate(
                        " ".join(df_raw["review"].astype(str).tolist())
                    )
                    st.image(wc.to_array(), use_container_width=True, caption="Word cloud (sample corpus)")
                except Exception:
                    pass

        if st.session_state.history:
            hist = pd.DataFrame(st.session_state.history)
            if "label" in hist.columns and "time" in hist.columns:
                hist["time"] = pd.to_datetime(hist["time"], errors="coerce")
                hist = hist.dropna(subset=["time"])
                if not hist.empty:
                    hist["bucket"] = hist["time"].dt.floor("min")
                    trend = hist.groupby(["bucket", "label"]).size().reset_index(name="count")
                    if not trend.empty:
                        fig_t = px.line(
                            trend,
                            x="bucket",
                            y="count",
                            color="label",
                            title="Recent session sentiment activity (per minute bucket)",
                        )
                        st.plotly_chart(fig_t, use_container_width=True)
        return

    if page == "Insights & history":
        st.subheader("Stored reviews and aggregate insights")
        st.caption(f"Persistence file: `{settings.data_dir / 'review_history.sqlite3'}`")
        init_db(settings)
        insights = aggregate_insights(settings)
        if insights.get("count", 0) == 0:
            st.info("No stored reviews yet. Use **Analyze → Single review → Predict** to save each run to the database.")
        else:
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("Reviews stored", f"{insights['count']}")
            c1.metric("Avg positive %", f"{insights['avg_positive_pct']:.1f}")
            c2.metric("Avg negative %", f"{insights['avg_negative_pct']:.1f}")
            c3.metric("Avg neutral %", f"{insights['avg_neutral_pct']:.1f}")

            df_hist = load_reviews_df(limit=800, settings=settings)
            if not df_hist.empty and "created_at" in df_hist.columns:
                df_time = df_hist.copy()
                df_time["created_at"] = pd.to_datetime(df_time["created_at"], errors="coerce")
                df_time = df_time.dropna(subset=["created_at"])
                if not df_time.empty:
                    fig_time = px.histogram(
                        df_time,
                        x="created_at",
                        nbinsx=30,
                        title="When reviews were analyzed (stored runs)",
                    )
                    st.plotly_chart(fig_time, use_container_width=True)

            lc = insights.get("label_counts") or {}
            if lc:
                fig_lab = px.pie(
                    names=list(lc.keys()),
                    values=list(lc.values()),
                    title="Overall predicted label mix (stored reviews)",
                )
                st.plotly_chart(fig_lab, use_container_width=True)

            am = insights.get("aspect_mention_counts") or {}
            if am:
                fig_asp = px.bar(
                    x=list(am.keys()),
                    y=list(am.values()),
                    labels={"x": "Aspect", "y": "Mentions"},
                    title="How often each aspect appeared in stored reviews",
                )
                st.plotly_chart(fig_asp, use_container_width=True)

            ap = insights.get("aspect_avg_positive_pct") or {}
            if ap:
                fig_pos = px.bar(
                    x=list(ap.keys()),
                    y=list(ap.values()),
                    labels={"x": "Aspect", "y": "Avg positive %"},
                    title="Average positive % when aspect is mentioned (per-aspect tone)",
                )
                st.plotly_chart(fig_pos, use_container_width=True)

            if not df_hist.empty:
                show_cols = [
                    c
                    for c in [
                        "created_at",
                        "model",
                        "overall_label",
                        "positive_pct",
                        "negative_pct",
                        "neutral_pct",
                        "text",
                    ]
                    if c in df_hist.columns
                ]
                st.subheader("Latest stored reviews")
                st.dataframe(df_hist[show_cols].head(100), use_container_width=True, hide_index=True)
                try:
                    blob = " ".join(df_hist["text"].astype(str).tolist()[:400])
                    if blob.strip():
                        wc = WordCloud(width=1000, height=420, background_color=None, mode="RGBA").generate(blob)
                        st.image(wc.to_array(), use_container_width=True, caption="Word cloud — stored reviews (recent window)")
                except Exception:
                    pass

        st.divider()
        ex1, ex2 = st.columns(2)
        with ex1:
            full_df = load_reviews_df(settings=settings)
            if not full_df.empty:
                csv_bytes = full_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download all stored reviews (CSV)",
                    data=csv_bytes,
                    file_name="stored_review_history.csv",
                    mime="text/csv",
                )
        with ex2:
            if st.button("Clear all stored reviews", type="secondary"):
                clear_all(settings)
                st.success("Stored review history cleared.")
                st.rerun()
        return

    if page == "Bulk CSV":
        st.subheader("Bulk review scoring")
        up = st.file_uploader("Upload CSV with columns `review` and `sentiment` (sentiment optional)", type=["csv"])
        choice = st.selectbox("Model for bulk run", model_options, key="bulk_model")
        ml_sel = None
        if choice.startswith("ML |"):
            if not ml_paths:
                st.error("Train ML models first.")
                return
            idx = ml_labels.index(choice)
            ml_sel = str(ml_paths[idx])
        if st.button("Run batch", type="primary") and up is not None:
            df_in = pd.read_csv(up)
            if "review" not in df_in.columns:
                st.error("CSV must include a `review` column.")
            else:
                preds = []
                confs = []
                for row in df_in["review"].astype(str):
                    res = run_prediction(row, choice, ml_sel)
                    preds.append(res.get("label"))
                    confs.append(res.get("confidence", 0.0))
                df_out = df_in.copy()
                df_out["pred_label"] = preds
                df_out["confidence"] = confs
                st.dataframe(df_out.head(50), use_container_width=True)
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        return

    tab_single, tab_compare = st.tabs(["Single review", "Model arena (compare all)"])

    with tab_compare:
        st.caption("Runs every available ML bundle plus optional LSTM/BERT on the same text.")
        t_compare = st.text_area("Review text (compare)", height=160, placeholder="Paste a product review...", key="cmp")
        if st.button("Compare all models", type="primary", key="cmp_btn") and t_compare.strip():
            rows = compare_models(t_compare.strip(), settings)
            if not rows:
                st.warning("No trained models yet. Train ML (`train-ml`) and/or deep models first.")
            else:
                cdf = pd.DataFrame(rows)
                st.dataframe(cdf, use_container_width=True, hide_index=True)
                figc = px.bar(
                    cdf,
                    x="model",
                    y="confidence",
                    color="label",
                    title="Confidence by model",
                )
                st.plotly_chart(figc, use_container_width=True)

    with tab_single:
        col_left, col_right = st.columns((1.1, 1))
        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            text = st.text_area("Review text", height=200, placeholder="Paste a product review...")
            model_choice = st.selectbox("Model", model_options)
            ml_path = None
            if model_choice.startswith("ML |"):
                if ml_paths:
                    ml_path = str(ml_paths[ml_labels.index(model_choice)])
                else:
                    st.warning("No ML models available. Train with `python main.py train-ml`.")
            use_lime = st.checkbox("Explain with LIME (ML pipelines only, slower)", value=False)
            run = st.button("Predict", type="primary")
            clear = st.button("Clear session table (does not delete stored DB)")
            if clear:
                st.session_state.history = []
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.caption("Confidence distribution for the last prediction appears below.")
            st.markdown("</div>", unsafe_allow_html=True)

        if run and text.strip():
            result = run_prediction(text, model_choice, ml_path)
            proc = get_preprocessor().transform(text)
            highlights = keyword_highlights(proc)
            scores = result.get("scores") or {}
            pct = scores_to_percentages(scores)

            def _predict_fn(chunk: str) -> dict:
                return run_prediction(chunk, model_choice, ml_path)

            with st.spinner("Aspect-based sentiment (scoring sentences that mention each aspect)..."):
                aspect_rows = aspect_based_sentiment(text.strip(), _predict_fn)

            report = build_prediction_report(text, model_choice, result, pct, aspect_rows)

            try:
                save_review(
                    text.strip(),
                    model_choice,
                    str(result.get("label", "")),
                    float(pct["positive_pct"]),
                    float(pct["negative_pct"]),
                    float(pct["neutral_pct"]),
                    aspect_rows,
                    scores,
                    settings,
                )
            except Exception as exc:
                st.caption(f"Note: could not persist review ({exc}).")

            add_history(
                {
                    "time": datetime.utcnow().isoformat(),
                    "model": model_choice,
                    "text": text[:400],
                    "label": result.get("label"),
                    "confidence": result.get("confidence"),
                    "positive_pct": pct["positive_pct"],
                    "negative_pct": pct["negative_pct"],
                    "neutral_pct": pct["neutral_pct"],
                }
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", str(result.get("label")).title())
            c2.metric("Top-class confidence", f"{float(result.get('confidence', 0))*100:.1f}%")
            c3.metric(
                "Model",
                model_choice.split("|")[-1].strip() if "|" in model_choice else model_choice,
            )

            st.subheader("Overall sentiment percentages")
            p1, p2, p3 = st.columns(3)
            p1.metric("Positive", f"{pct['positive_pct']:.1f}%")
            p2.metric("Negative", f"{pct['negative_pct']:.1f}%")
            p3.metric("Neutral", f"{pct['neutral_pct']:.1f}%")
            fig_mix = px.pie(
                names=["Positive", "Negative", "Neutral"],
                values=[pct["positive_pct"], pct["negative_pct"], pct["neutral_pct"]],
                title="Overall polarity mix (from model class scores)",
                hole=0.35,
            )
            st.plotly_chart(fig_mix, use_container_width=True)

            st.subheader("Aspect-based sentiment")
            st.caption(
                "Each aspect uses keyword cues, then averages sentiment on matching sentences. "
                "Extend `ASPECT_LEXICON` in `app/aspect_analysis.py` for your domain."
            )
            disp_rows = []
            for row in aspect_rows:
                if not row.get("mentioned"):
                    disp_rows.append(
                        {
                            "Aspect": row["aspect"],
                            "Mentioned": "No",
                            "Dominant": "—",
                            "Positive %": "—",
                            "Negative %": "—",
                            "Neutral %": "—",
                        }
                    )
                else:
                    disp_rows.append(
                        {
                            "Aspect": row["aspect"],
                            "Mentioned": "Yes",
                            "Dominant": str(row.get("dominant_label") or "—").title(),
                            "Positive %": f"{row.get('positive_pct', 0):.1f}" if row.get("positive_pct") is not None else "—",
                            "Negative %": f"{row.get('negative_pct', 0):.1f}" if row.get("negative_pct") is not None else "—",
                            "Neutral %": f"{row.get('neutral_pct', 0):.1f}" if row.get("neutral_pct") is not None else "—",
                        }
                    )
            st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, hide_index=True)
            with st.expander("View text snippets used per aspect"):
                for row in aspect_rows:
                    if row.get("mentioned") and row.get("snippets"):
                        st.markdown(f"**{row['aspect']}**")
                        for sn in row["snippets"]:
                            st.write(f"- {sn}")

            if scores:
                fig = px.bar(
                    x=list(scores.keys()),
                    y=list(scores.values()),
                    labels={"x": "Class", "y": "Score"},
                    title="Raw class scores (model output)",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.write("**Explainability (keywords from preprocessing):**", ", ".join(highlights) or "n/a")
            st.write("**Processed snippet:**", result.get("processed_preview", proc[:280]))

            if use_lime and model_choice.startswith("ML |") and ml_path:
                from utils.explainability import explain_with_lime

                bundle = load_cached_ml(ml_path)
                pipe = bundle["pipeline"]
                le = bundle.get("label_encoder")
                class_names = None
                if le is not None:
                    class_names = [str(x) for x in le.classes_.tolist()]
                with st.spinner("Computing LIME explanation..."):
                    lime_res = explain_with_lime(
                        text,
                        pipe,
                        get_preprocessor(),
                        class_names=class_names,
                        num_samples=1500,
                    )
                if lime_res:
                    st.success(f"LIME top label: **{lime_res['label_name']}**")
                    st.table(pd.DataFrame(lime_res["weights"], columns=["feature", "weight"]))
                else:
                    st.info("Install `lime` (`pip install lime`) to enable local explanations.")

            st.download_button(
                "Download prediction JSON",
                data=json.dumps(report, indent=2, default=str).encode("utf-8"),
                file_name="prediction_report.json",
                mime="application/json",
            )

        if st.session_state.history:
            st.subheader("Recent predictions")
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)


if __name__ == "__main__":
    main()
