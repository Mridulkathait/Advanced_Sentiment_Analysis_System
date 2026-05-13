"""Advanced NLP preprocessing pipeline."""

from __future__ import annotations

import html
import re
import string
from typing import Iterable, List, Optional

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lazy NLTK data download handled on first use
_nltk_ready = False


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk

    for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
        try:
            if pkg in {"stopwords", "wordnet", "omw-1.4"}:
                nltk.data.find(f"corpora/{pkg}")
            else:
                nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    _nltk_ready = True


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class TextPreprocessor:
    """Configurable text cleaning and normalization."""

    def __init__(
        self,
        lowercase: bool = True,
        strip_html: bool = True,
        strip_urls: bool = True,
        remove_punctuation: bool = True,
        remove_emojis: bool = True,
        remove_numbers: bool = True,
        normalize_whitespace: bool = True,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        use_lemmatization: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.strip_html = strip_html
        self.strip_urls = strip_urls
        self.remove_punctuation = remove_punctuation
        self.remove_emojis = remove_emojis
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self._stemmer: Optional[PorterStemmer] = None
        self._lemmatizer: Optional[WordNetLemmatizer] = None
        self._stop: Optional[set] = None

    def _lazy_init(self) -> None:
        _ensure_nltk()
        if self._stop is None and self.remove_stopwords:
            self._stop = set(stopwords.words("english"))
        if self.use_stemming and self._stemmer is None:
            self._stemmer = PorterStemmer()
        if self.use_lemmatization and self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _regex_clean(text: str) -> str:
        text = html.unescape(text)
        return text

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    def _remove_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _remove_special(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = self._regex_clean(text)
        if self.strip_html:
            text = self._remove_html(text)
        if self.strip_urls:
            text = self._remove_urls(text)
        if self.remove_emojis:
            text = EMOJI_PATTERN.sub(" ", text)
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        if self.remove_punctuation:
            table = str.maketrans("", "", string.punctuation)
            text = text.translate(table)
        else:
            text = self._remove_special(text)
        if self.lowercase:
            text = text.lower()
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_normalize(self, text: str) -> List[str]:
        self._lazy_init()
        tokens = word_tokenize(text)
        out: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            if self.remove_stopwords and self._stop and tok in self._stop:
                continue
            if len(tok) < 2:
                continue
            w = tok
            if self.use_lemmatization and self._lemmatizer:
                w = self._lemmatizer.lemmatize(w)
            if self.use_stemming and self._stemmer:
                w = self._stemmer.stem(w)
            out.append(w)
        return out

    def transform(self, text: str) -> str:
        cleaned = self.clean(text)
        toks = self.tokenize_and_normalize(cleaned)
        return " ".join(toks)

    def transform_series(self, series: Iterable[str]) -> pd.Series:
        return pd.Series(series).astype(str).map(self.transform)

    def transform_batch(self, texts: List[str]) -> List[str]:
        return [self.transform(t) for t in texts]
