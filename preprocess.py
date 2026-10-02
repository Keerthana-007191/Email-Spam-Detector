"""Email-aware text preprocessing, packaged as a scikit-learn transformer."""
from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Any, List

from sklearn.base import BaseEstimator, TransformerMixin

URL_TOKEN = "urltoken"
EMAIL_TOKEN = "emailtoken"
NUM_TOKEN = "numtoken"
DOLLAR_TOKEN = "dollartoken"
EXCLAIM_TOKEN = "exclaimtoken"

# Words that carry meaning and must survive stopword removal.
NEGATIONS = frozenset({
    "no", "nor", "not", "never", "none", "neither",
    "nobody", "nothing", "nowhere", "cannot",
})

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S)
_TAG = re.compile(r"<[/!]?[a-z][^>]{0,1000}>")
_URL = re.compile(r"(?<![@\w])(?:https?://|ftp://|www\.)[^\s<>\"']+")
_EMAIL = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CANT = re.compile(r"\bcan\s*'\s*t\b")
_WONT = re.compile(r"\bwon\s*'\s*t\b")
_NT = re.compile(r"\b(\w+?)\s*n\s*'\s*t\b")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """English stopwords minus negations. Falls back to scikit-learn's list offline."""
    try:
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words = set(ENGLISH_STOP_WORDS)
    return frozenset(words - NEGATIONS)


def _to_str(value: Any) -> str:
    """Coerce any input (None, NaN, bytes, numbers) into a string."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value)


def _tag_to_text(match: re.Match) -> str:
    """Replace an HTML tag with a space, but keep any URL found inside it."""
    urls = _URL.findall(match.group(0))
    return " " + " ".join(urls) + " "


def clean_text(text: Any, remove_stopwords: bool = True,
               mask_numbers: bool = True, max_chars: int = 20000) -> str:
    """Clean one email into a space-separated string of tokens."""
    text = _to_str(text)
    if not text.strip():
        return ""

    text = text[:max_chars].lower()
    text = html.unescape(text).replace("\u2019", "'")

    # HTML
    text = _COMMENT.sub(" ", text)
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _TAG.sub(_tag_to_text, text)

    # Email-specific patterns -> fixed tokens
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _EMAIL.sub(f" {EMAIL_TOKEN} ", text)
    text = text.replace("$", f" {DOLLAR_TOKEN} ").replace("!", f" {EXCLAIM_TOKEN} ")
    if mask_numbers:
        text = _NUMBER.sub(f" {NUM_TOKEN} ", text)

    # Contractions, so negations survive punctuation removal
    text = _CANT.sub(" cannot ", text)
    text = _WONT.sub(" will not ", text)
    text = _NT.sub(r"\1 not", text)

    # Punctuation -> space, then tokenise on whitespace
    tokens = [t for t in _NON_ALNUM.sub(" ", text).split() if len(t) > 1]
    if remove_stopwords:
        sw = get_stopwords()
        tokens = [t for t in tokens if t not in sw]
    return " ".join(tokens)


class EmailPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless transformer: list/Series of raw emails -> list of cleaned strings."""

    def __init__(self, remove_stopwords: bool = True,
                 mask_numbers: bool = True, max_chars: int = 20000):
        self.remove_stopwords = remove_stopwords
        self.mask_numbers = mask_numbers
        self.max_chars = max_chars

    def fit(self, X, y=None):
        return self  # nothing is learned, so no leakage is possible

    def transform(self, X) -> List[str]:
        if isinstance(X, (str, bytes)):
            X = [X]
        return [clean_text(t, self.remove_stopwords, self.mask_numbers, self.max_chars)
                for t in X]


if __name__ == "__main__":
    samples = [
        "WIN $1000 NOW!!! Click <a href=\"http://bad.example.com/x\">here</a>",
        "Hi John, I don't think the meeting on 12/05 works. Mail me at john@enron.com",
        "",
    ]
    for s in samples:
        print(repr(s), "->", repr(clean_text(s)))