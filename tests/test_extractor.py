"""
test_extractor.py
=================
Tests for the PDF extraction module.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.information_extractor import (
    extract_amount, extract_percentage, extract_income,
    extract_deadline, extract_documents, extract_scholarship_name,
    extract_organization, extract_eligibility, extract_all,
)


# ---------------------------------------------------------------------------
# Sample texts for testing
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
ABC Education Foundation Scholarship 2026

The scholarship is available to undergraduate students.
Applicants must have obtained at least 75% marks.
Annual family income must not exceed Rs. 4,00,000.
Selected students will receive Rs. 50,000 per year.
The last date for application is 30 September 2026.

Applicants must submit:
Aadhaar Card
Previous Year Marksheet
Income Certificate
Bonafide Certificate
Bank Account Details

Applications must be submitted through the official online application portal.
"""

EMPTY_TEXT = ""

SHORT_TEXT = "Short document."

MISSING_FIELDS_TEXT = """
This is a scholarship document without specific amounts or dates.
Students must be eligible and meet the eligibility criteria.
Please contact the organization for more information.
"""


# ---------------------------------------------------------------------------
# Tests: Scholarship Name
# ---------------------------------------------------------------------------

def test_scholarship_name_found():
    """Should find 'ABC Education Foundation Scholarship 2026' in text."""
    result = extract_scholarship_name(SAMPLE_TEXT)
    assert "scholarship" in result.lower() or "ABC" in result, \
        f"Expected scholarship name, got: {result}"


def test_scholarship_name_empty():
    result = extract_scholarship_name(EMPTY_TEXT)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Amount Extraction
# ---------------------------------------------------------------------------

def test_amount_found():
    result = extract_amount(SAMPLE_TEXT)
    assert "not found" not in result.lower(), f"Expected amount, got: {result}"
    assert any(sym in result for sym in ["₹", "Rs", "50,000", "50000"]), \
        f"Expected currency in result, got: {result}"


def test_amount_empty():
    result = extract_amount(EMPTY_TEXT)
    assert "not found" in result.lower()


def test_amount_missing():
    result = extract_amount(MISSING_FIELDS_TEXT)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Percentage Extraction
# ---------------------------------------------------------------------------

def test_percentage_found():
    result = extract_percentage(SAMPLE_TEXT)
    assert "%" in result, f"Expected percentage symbol, got: {result}"
    assert "75" in result, f"Expected 75%, got: {result}"


def test_percentage_various_formats():
    texts = [
        "Minimum 60% marks required.",
        "At least 70 percent in qualifying exam.",
        "Students must score minimum percentage of 65%.",
    ]
    for text in texts:
        result = extract_percentage(text)
        assert "%" in result and "not found" not in result.lower(), \
            f"Failed for: {text}\nGot: {result}"


def test_percentage_empty():
    result = extract_percentage(EMPTY_TEXT)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Income Extraction
# ---------------------------------------------------------------------------

def test_income_found():
    result = extract_income(SAMPLE_TEXT)
    assert "not found" not in result.lower(), f"Expected income, got: {result}"
    assert "4" in result or "income" in result.lower(), f"Got: {result}"


def test_income_empty():
    result = extract_income(EMPTY_TEXT)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Deadline Extraction
# ---------------------------------------------------------------------------

def test_deadline_found():
    result = extract_deadline(SAMPLE_TEXT)
    assert "not found" not in result.lower(), f"Expected deadline, got: {result}"
    # Should contain "September" or "2026" or "30"
    assert any(kw in result for kw in ["September", "2026", "30"]), \
        f"Expected date components, got: {result}"


def test_deadline_various_formats():
    texts = [
        ("30/09/2026 is the last date", ["30", "09", "2026"]),
        ("Apply before 30-09-2026", ["30", "09", "2026"]),
        ("Deadline: September 30, 2026", ["September", "30", "2026"]),
    ]
    for text, expected_parts in texts:
        result = extract_deadline(text)
        assert "not found" not in result.lower() or any(p in result for p in expected_parts), \
            f"Failed for: {text}\nGot: {result}"


def test_deadline_empty():
    result = extract_deadline(EMPTY_TEXT)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Document Extraction
# ---------------------------------------------------------------------------

def test_documents_found():
    docs = extract_documents(SAMPLE_TEXT)
    assert isinstance(docs, list)
    assert len(docs) > 0, "Expected at least one document to be found"


def test_documents_aadhaar():
    docs = extract_documents(SAMPLE_TEXT)
    doc_names_lower = [d.lower() for d in docs]
    assert any("aadhaar" in d for d in doc_names_lower), \
        f"Expected Aadhaar in documents, got: {docs}"


def test_documents_empty():
    docs = extract_documents(EMPTY_TEXT)
    assert docs == []


def test_documents_not_assumed():
    """Documents not mentioned should NOT be returned."""
    text_without_docs = "This scholarship is for all undergraduate students."
    docs = extract_documents(text_without_docs)
    assert len(docs) == 0, \
        f"Should not find documents in text without them. Got: {docs}"


# ---------------------------------------------------------------------------
# Tests: Extract All
# ---------------------------------------------------------------------------

def test_extract_all_returns_all_keys():
    result = extract_all(SAMPLE_TEXT)
    expected_keys = [
        "scholarship_name", "organization", "amount",
        "percentage_requirement", "income_requirement",
        "deadline", "eligibility", "documents", "application_process"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_extract_all_empty_text():
    result = extract_all(EMPTY_TEXT)
    assert isinstance(result, dict)
    # Should not raise an exception
    assert "scholarship_name" in result


if __name__ == "__main__":
    print("Running information extractor tests...\n")

    tests = [
        test_scholarship_name_found,
        test_scholarship_name_empty,
        test_amount_found,
        test_amount_empty,
        test_amount_missing,
        test_percentage_found,
        test_percentage_various_formats,
        test_percentage_empty,
        test_income_found,
        test_income_empty,
        test_deadline_found,
        test_deadline_various_formats,
        test_deadline_empty,
        test_documents_found,
        test_documents_aadhaar,
        test_documents_empty,
        test_documents_not_assumed,
        test_extract_all_returns_all_keys,
        test_extract_all_empty_text,
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
