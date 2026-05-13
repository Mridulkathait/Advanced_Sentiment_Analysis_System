"""Lightweight caching helpers for Streamlit and batch inference."""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def stable_hash(text: str, max_len: int = 10_000) -> str:
    """Deterministic short hash for cache keys."""
    payload = text[:max_len].encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def optional_streamlit_cache(func: F) -> F:
    """Apply st.cache_resource if Streamlit is available."""
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)  # type: ignore[return-value]
    except Exception:
        return func
