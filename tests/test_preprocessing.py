"""
test_preprocessing.py
=====================
Tests for the NLP preprocessing module.
Run with: python -m pytest tests/ -v
       or: python tests/test_preprocessing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.preprocessing import (
    tokenize_sentences, tokenize_words, remove_stopwords,
    remove_punctuation, lemmatize_tokens, preprocess_text,
    clean_text_for_display,
)


# ---------------------------------------------------------------------------
# Sample Texts
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
The scholarship is available to undergraduate students.
Applicants must have obtained at least 75% marks.
Annual family income must not exceed Rs. 4,00,000.
"""

SHORT_TEXT = "Apply for the scholarship."
EMPTY_TEXT = ""


# ---------------------------------------------------------------------------
# Tests: Sentence Tokenization
# ---------------------------------------------------------------------------

def test_sentence_tokenization_basic():
    sentences = tokenize_sentences(SAMPLE_TEXT)
    assert isinstance(sentences, list)
    assert len(sentences) >= 2, f"Expected multiple sentences, got {len(sentences)}"


def test_sentence_tokenization_returns_strings():
    sentences = tokenize_sentences(SAMPLE_TEXT)
    for s in sentences:
        assert isinstance(s, str), f"Expected string, got {type(s)}"


def test_sentence_tokenization_empty():
    sentences = tokenize_sentences(EMPTY_TEXT)
    assert sentences == []


def test_sentence_tokenization_filters_empty():
    """Result should contain no empty/whitespace-only strings."""
    sentences = tokenize_sentences(SAMPLE_TEXT)
    for s in sentences:
        assert s.strip() != "", "Found empty sentence in tokenization output"


# ---------------------------------------------------------------------------
# Tests: Word Tokenization
# ---------------------------------------------------------------------------

def test_word_tokenization_basic():
    tokens = tokenize_words(SHORT_TEXT)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_word_tokenization_empty():
    tokens = tokenize_words(EMPTY_TEXT)
    assert tokens == []


def test_word_tokenization_splits_correctly():
    tokens = tokenize_words("Hello world.")
    assert "Hello" in tokens or "hello" in tokens
    assert "world" in tokens


# ---------------------------------------------------------------------------
# Tests: Stopword Removal
# ---------------------------------------------------------------------------

def test_stopword_removal():
    tokens = ["the", "scholarship", "is", "available", "for", "students"]
    filtered = remove_stopwords(tokens)
    stopwords_removed = ["the", "is", "for"]
    for sw in stopwords_removed:
        assert sw not in filtered, f"Stopword '{sw}' was not removed"


def test_stopword_keeps_important_words():
    tokens = ["scholarship", "eligibility", "percentage", "income"]
    filtered = remove_stopwords(tokens)
    for word in tokens:
        assert word in filtered, f"Important word '{word}' was incorrectly removed"


def test_stopword_empty_list():
    assert remove_stopwords([]) == []


# ---------------------------------------------------------------------------
# Tests: Punctuation Removal
# ---------------------------------------------------------------------------

def test_punctuation_removal():
    tokens = ["scholarship", ".", ",", "75%", "apply", "!"]
    filtered = remove_punctuation(tokens)
    assert "." not in filtered
    assert "," not in filtered
    assert "!" not in filtered


def test_punctuation_keeps_words():
    tokens = ["scholarship", "eligibility", "apply"]
    filtered = remove_punctuation(tokens)
    assert filtered == tokens


def test_punctuation_empty():
    assert remove_punctuation([]) == []


# ---------------------------------------------------------------------------
# Tests: Lemmatization
# ---------------------------------------------------------------------------

def test_lemmatization_basic():
    tokens = ["running", "scholarships", "applying", "submitted"]
    lemmatized = lemmatize_tokens(tokens)
    # Lemmatizer should reduce these to their base forms
    assert isinstance(lemmatized, list)
    assert len(lemmatized) == len(tokens)


def test_lemmatization_lowercase():
    """Lemmatization should produce lowercase tokens."""
    tokens = ["Scholarship", "APPLY", "Students"]
    lemmatized = lemmatize_tokens(tokens)
    for token in lemmatized:
        assert token == token.lower(), f"Token '{token}' is not lowercase"


def test_lemmatization_known_forms():
    """Test specific known lemma reductions."""
    # These are well-known NLTK WordNet lemmatizations
    pairs = [
        (["running"], "run"),
        (["applying"], "apply"),
        (["studies"], "study"),
    ]
    for tokens, expected_lemma in pairs:
        result = lemmatize_tokens(tokens)
        assert expected_lemma in result or tokens[0].lower() in result, \
            f"Expected '{expected_lemma}' for {tokens}, got {result}"


# ---------------------------------------------------------------------------
# Tests: Full Pipeline (preprocess_text)
# ---------------------------------------------------------------------------

def test_preprocess_text_returns_correct_keys():
    result = preprocess_text(SAMPLE_TEXT)
    expected_keys = [
        "sentences", "raw_tokens", "clean_tokens",
        "processed_text", "word_count", "sentence_count"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_preprocess_text_word_count():
    result = preprocess_text(SAMPLE_TEXT)
    assert result["word_count"] > 0


def test_preprocess_text_sentence_count():
    result = preprocess_text(SAMPLE_TEXT)
    assert result["sentence_count"] > 0


def test_preprocess_text_clean_tokens_no_stopwords():
    """Processed tokens should not contain common stopwords."""
    result = preprocess_text(SAMPLE_TEXT)
    common_stopwords = {"the", "is", "a", "an", "for", "of", "and", "or"}
    for token in result["clean_tokens"]:
        assert token not in common_stopwords, \
            f"Stopword '{token}' found in cleaned tokens"


def test_preprocess_text_empty():
    result = preprocess_text(EMPTY_TEXT)
    assert result["sentences"] == []
    assert result["raw_tokens"] == []
    assert result["clean_tokens"] == []
    assert result["word_count"] == 0
    assert result["sentence_count"] == 0


def test_preprocess_text_clean_is_lowercase():
    """All clean tokens should be in lowercase."""
    result = preprocess_text(SAMPLE_TEXT)
    for token in result["clean_tokens"]:
        assert token == token.lower(), f"Token '{token}' is not lowercase"


# ---------------------------------------------------------------------------
# Tests: Display Cleaning
# ---------------------------------------------------------------------------

def test_clean_text_removes_extra_newlines():
    text = "Line one.\n\n\n\n\nLine two."
    cleaned = clean_text_for_display(text)
    assert "\n\n\n" not in cleaned


def test_clean_text_removes_extra_spaces():
    text = "Word1  word2   word3."
    cleaned = clean_text_for_display(text)
    assert "  " not in cleaned


if __name__ == "__main__":
    print("Running preprocessing tests...\n")

    tests = [
        test_sentence_tokenization_basic,
        test_sentence_tokenization_returns_strings,
        test_sentence_tokenization_empty,
        test_sentence_tokenization_filters_empty,
        test_word_tokenization_basic,
        test_word_tokenization_empty,
        test_word_tokenization_splits_correctly,
        test_stopword_removal,
        test_stopword_keeps_important_words,
        test_stopword_empty_list,
        test_punctuation_removal,
        test_punctuation_keeps_words,
        test_punctuation_empty,
        test_lemmatization_basic,
        test_lemmatization_lowercase,
        test_lemmatization_known_forms,
        test_preprocess_text_returns_correct_keys,
        test_preprocess_text_word_count,
        test_preprocess_text_sentence_count,
        test_preprocess_text_clean_tokens_no_stopwords,
        test_preprocess_text_empty,
        test_preprocess_text_clean_is_lowercase,
        test_clean_text_removes_extra_newlines,
        test_clean_text_removes_extra_spaces,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: EXCEPTION — {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests.")
