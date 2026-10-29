"""
preprocessing.py
================
NLP Text Preprocessing Pipeline using NLTK.

NLP Concepts Covered:
- Sentence Tokenization  : Breaking text into sentences
- Word Tokenization      : Breaking sentences into individual words/tokens
- Lowercase Conversion   : Normalisation so "Scholarship" == "scholarship"
- Stopword Removal       : Removing common words (the, is, a) that add no meaning
- Punctuation Removal    : Cleaning non-alphanumeric characters
- Lemmatization          : Reducing words to their base/dictionary form
                           (e.g., "running" → "run", "scholarships" → "scholarship")
"""

import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK Resource Bootstrapping
# ---------------------------------------------------------------------------
# We attempt to download required NLTK data if it is not already present.
# This prevents the application from crashing on first run.

_REQUIRED_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_resources():
    """Download any missing NLTK resources silently."""
    for resource_path, resource_name in _REQUIRED_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass  # If download fails, we handle gracefully later


# Run resource check on module import
ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Module-level Globals (initialised once for performance)
# ---------------------------------------------------------------------------
try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    _STOP_WORDS = set()

_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Core Preprocessing Functions
# ---------------------------------------------------------------------------


def tokenize_sentences(text: str) -> list:
    """
    Sentence Tokenization
    ---------------------
    Split the raw document text into individual sentences using NLTK's
    Punkt sentence tokenizer, which is trained on English punctuation patterns.

    Parameters
    ----------
    text : str — raw document text

    Returns
    -------
    list of str — each element is one sentence
    """
    if not text or not text.strip():
        return []

    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback: split on period/newline if NLTK tokenizer unavailable
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    # Filter out empty/whitespace-only sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize_words(text: str) -> list:
    """
    Word Tokenization
    -----------------
    Split text into individual word tokens using NLTK's word_tokenize.
    This handles contractions, punctuation, and special characters better
    than a simple str.split().

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str — individual tokens
    """
    if not text:
        return []

    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback to simple whitespace split
        tokens = text.split()

    return tokens


def remove_stopwords(tokens: list) -> list:
    """
    Stopword Removal
    ----------------
    Remove common English words (stopwords) that occur frequently
    but carry little semantic meaning, e.g.: the, is, at, which, on.

    This reduces noise and helps the TF-IDF model focus on meaningful terms.

    Parameters
    ----------
    tokens : list of str

    Returns
    -------
    list of str — tokens without stopwords
    """
    return [t for t in tokens if t.lower() not in _STOP_WORDS]


def remove_punctuation(tokens: list) -> list:
    """
    Punctuation Removal
    -------------------
    Remove tokens that consist entirely of punctuation characters.
    Keeps alphanumeric tokens (words and numbers).

    Parameters
    ----------
    tokens : list of str

    Returns
    -------
    list of str — tokens without punctuation-only entries
    """
    return [t for t in tokens if t.isalnum()]


def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatization
    -------------
    Convert each word to its canonical (dictionary/base) form using
    NLTK's WordNetLemmatizer.

    Examples:
        "applying"    → "apply"
        "scholarships" → "scholarship"
        "submitted"   → "submit"
        "eligible"    → "eligible"

    This helps the TF-IDF model match related word forms as the same term.

    Parameters
    ----------
    tokens : list of str

    Returns
    -------
    list of str — lemmatized tokens
    """
    return [_LEMMATIZER.lemmatize(t.lower()) for t in tokens]


def preprocess_text(text: str) -> dict:
    """
    Full NLP Preprocessing Pipeline
    --------------------------------
    Runs the complete preprocessing pipeline on the input text.

    Pipeline:
        Raw Text
          ↓ Sentence Tokenization
          ↓ Word Tokenization
          ↓ Lowercase Conversion
          ↓ Stopword Removal
          ↓ Punctuation Removal
          ↓ Lemmatization
          ↓ Cleaned Tokens

    Parameters
    ----------
    text : str — raw extracted text from the PDF

    Returns
    -------
    dict with keys:
        'sentences'       : list of str — sentence tokens (for summarizer)
        'raw_tokens'      : list of str — all word tokens before cleaning
        'clean_tokens'    : list of str — fully processed tokens
        'processed_text'  : str — clean tokens joined as a string
        'word_count'      : int — total words in raw text
        'sentence_count'  : int — total sentences
    """
    if not text or not text.strip():
        return {
            "sentences": [],
            "raw_tokens": [],
            "clean_tokens": [],
            "processed_text": "",
            "word_count": 0,
            "sentence_count": 0,
        }

    # Step 1: Sentence tokenization
    sentences = tokenize_sentences(text)

    # Step 2: Word tokenization
    raw_tokens = tokenize_words(text)

    # Step 3: Lowercase conversion (done inside lemmatize_tokens)
    tokens = raw_tokens[:]

    # Step 4: Remove stopwords
    tokens = remove_stopwords(tokens)

    # Step 5: Remove punctuation/special characters
    tokens = remove_punctuation(tokens)

    # Step 6: Lemmatization
    clean_tokens = lemmatize_tokens(tokens)

    return {
        "sentences": sentences,
        "raw_tokens": raw_tokens,
        "clean_tokens": clean_tokens,
        "processed_text": " ".join(clean_tokens),
        "word_count": len(raw_tokens),
        "sentence_count": len(sentences),
    }


def clean_text_for_display(text: str) -> str:
    """
    Light cleaning for display purposes (not for NLP models).
    Removes excessive whitespace and normalises newlines.

    Parameters
    ----------
    text : str

    Returns
    -------
    str — cleaned text suitable for display
    """
    # Collapse multiple whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
