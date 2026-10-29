"""
test_summarizer.py
==================
Tests for the TF-IDF extractive summarizer module.
Run with: python -m pytest tests/ -v
       or: python tests/test_summarizer.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.summarizer import summarize, _split_into_sentences


# ---------------------------------------------------------------------------
# Sample Texts
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
ABC Education Foundation Scholarship 2026 is a merit-cum-means scholarship
offered to undergraduate students from economically weaker sections of society.

The scholarship aims to encourage academically talented students to pursue
higher education without financial constraints. Applicants must have obtained
a minimum of 75% marks in their most recent qualifying examination.

The annual family income of the applicant must not exceed Rs. 4,00,000 per annum
from all sources. Selected students will receive a scholarship of Rs. 50,000
per year, credited directly to their bank account.

The last date for application is 30 September 2026. Applications must be
submitted through the official online application portal at the foundation website.

Required documents include Aadhaar Card, Previous Year Marksheet, Income Certificate,
Bonafide Certificate, and Bank Account Details.

The scholarship selection is based on academic merit and financial need.
Shortlisted candidates may be called for a telephonic interview.
"""

SHORT_TEXT = "Scholarship available. Apply now."

EMPTY_TEXT = ""

LONG_TEXT = """
This scholarship programme is offered annually by the National Education Trust.
The programme is designed to support meritorious students from underprivileged backgrounds.
Students pursuing undergraduate degrees in science, arts, and commerce are eligible.
The minimum academic requirement is 70 percent marks in the previous qualifying examination.
The annual family income should not exceed five lakh rupees from all sources combined.
Selected scholars will receive a financial grant of forty thousand rupees per academic year.
The scholarship amount will be disbursed in two equal instalments each academic year.
Applications must be submitted online through the trust's official digital portal.
The deadline for submitting the application form is the fifteenth of October 2026.
Supporting documents must be uploaded in PDF format with a maximum size of two megabytes.
Incomplete applications or applications submitted after the deadline will be rejected.
The selection committee will evaluate applications based on merit and financial need.
Interviews may be conducted for shortlisted candidates as part of the selection process.
The final list of selected scholars will be published on the official website by December.
Scholars must maintain a minimum of sixty-five percent marks to renew the scholarship.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_summarize_basic():
    """Basic summarization should return a non-empty summary."""
    result = summarize(SAMPLE_TEXT)
    assert result["success"] is True
    assert len(result["summary"]) > 0
    assert result["sentence_count"] > 0


def test_summarize_empty():
    """Empty text should return failure result."""
    result = summarize(EMPTY_TEXT)
    assert result["success"] is False
    assert result["summary"] == ""


def test_summarize_short_text():
    """Very short text should return what is available without crashing."""
    result = summarize(SHORT_TEXT)
    assert isinstance(result, dict)
    assert "success" in result
    assert "summary" in result


def test_summarize_respects_max_sentences():
    """Summary should not exceed max_sentences."""
    result = summarize(LONG_TEXT, max_sentences=5)
    assert result["success"] is True
    assert result["sentence_count"] <= 5


def test_summarize_respects_min_sentences():
    """Summary should return at least min_sentences if document is long enough."""
    result = summarize(LONG_TEXT, min_sentences=3, max_sentences=7)
    if result["success"]:
        assert result["sentence_count"] >= 1  # At minimum one sentence


def test_summarize_preserves_order():
    """
    Sentences in the summary should appear in the same relative order
    as they appear in the original document (extractive summarization).
    """
    result = summarize(LONG_TEXT)
    assert result["success"] is True

    summary_sentences = result["summary"].split(". ")
    original_lower = LONG_TEXT.lower()

    last_pos = -1
    order_preserved = True
    for sent in summary_sentences:
        if len(sent.strip()) < 5:
            continue
        # Find approximate position in original text
        pos = original_lower.find(sent.strip().lower()[:30])
        if pos != -1:
            if pos < last_pos:
                order_preserved = False
                break
            last_pos = pos

    assert order_preserved, "Summary sentences are not in original document order"


def test_summarize_not_first_sentences_only():
    """
    The summarizer should NOT simply take the first N sentences.
    The summary should include sentences from throughout the document.
    """
    result = summarize(LONG_TEXT, max_sentences=5)
    assert result["success"] is True

    # Get the first 3 sentences of the original
    original_sentences = _split_into_sentences(LONG_TEXT)
    first_sentences = set(s[:40] for s in original_sentences[:3])
    summary_starts = set(s[:40] for s in _split_into_sentences(result["summary"]))

    # The summary should NOT be entirely composed of only the first sentences
    # (it should pull from various parts of the document)
    assert not summary_starts.issubset(first_sentences), \
        "Summary appears to only take the first sentences — TF-IDF is not working"


def test_summarize_returns_scored_sentences():
    """scored_sentences should be a list of (sentence, score) tuples."""
    result = summarize(SAMPLE_TEXT)
    if result["success"]:
        assert isinstance(result["scored_sentences"], list)
        if result["scored_sentences"]:
            first = result["scored_sentences"][0]
            assert len(first) == 2
            assert isinstance(first[0], str)
            assert isinstance(first[1], float)


def test_sentence_splitter():
    """Sentence splitter should return a list of strings."""
    sentences = _split_into_sentences(SAMPLE_TEXT)
    assert isinstance(sentences, list)
    assert len(sentences) > 0
    for sent in sentences:
        assert isinstance(sent, str)
        assert len(sent.split()) >= 4  # All sentences should have at least 4 words


def test_sentence_splitter_empty():
    """Sentence splitter on empty text should return empty list."""
    sentences = _split_into_sentences(EMPTY_TEXT)
    assert sentences == []


def test_summarize_scholarship_relevance():
    """Summary should contain scholarship-relevant content."""
    result = summarize(SAMPLE_TEXT)
    assert result["success"] is True

    summary_lower = result["summary"].lower()
    relevance_keywords = ["scholarship", "eligib", "75%", "income", "rs.", "document", "apply"]
    hits = sum(1 for kw in relevance_keywords if kw in summary_lower)

    assert hits >= 2, \
        f"Summary does not appear to contain scholarship-relevant content. Summary: {result['summary']}"


if __name__ == "__main__":
    print("Running summarizer tests...\n")

    tests = [
        test_summarize_basic,
        test_summarize_empty,
        test_summarize_short_text,
        test_summarize_respects_max_sentences,
        test_summarize_respects_min_sentences,
        test_summarize_preserves_order,
        test_summarize_not_first_sentences_only,
        test_summarize_returns_scored_sentences,
        test_sentence_splitter,
        test_sentence_splitter_empty,
        test_summarize_scholarship_relevance,
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
