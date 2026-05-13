"""Feature extraction: BoW, TF-IDF, n-grams."""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

VectorizerKind = Literal["bow", "tfidf"]


def build_vectorizer(
    kind: VectorizerKind = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> CountVectorizer | TfidfVectorizer:
    common = dict(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    if kind == "bow":
        return CountVectorizer(**common)
    return TfidfVectorizer(sublinear_tf=sublinear_tf, **common)


def build_multi_ngram_union(max_features: int = 15000) -> FeatureUnion:
    """Separate branches for unigrams+bigrams and trigrams (combined sparse)."""
    uni_bi = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2, max_df=0.95)
    tri = TfidfVectorizer(ngram_range=(3, 3), max_features=max(1000, max_features // 4), min_df=2, max_df=0.98)
    return FeatureUnion([("uni_bi", uni_bi), ("tri", tri)])


def fit_transform_texts(
    train_texts: Iterable[str],
    test_texts: Optional[Iterable[str]] = None,
    kind: VectorizerKind = "tfidf",
    ngram_range: Tuple[int, int] = (1, 3),
    max_features: int = 20000,
):
    vec = build_vectorizer(kind=kind, ngram_range=ngram_range, max_features=max_features)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts) if test_texts is not None else None
    return X_train, X_test, vec
