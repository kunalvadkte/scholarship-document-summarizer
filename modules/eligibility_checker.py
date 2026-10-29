"""
eligibility_checker.py
======================
Rule-based Eligibility Checker.

NLP Concept:
  This module demonstrates rule-based NLP — a deterministic system that
  applies explicit logical rules to structured data, as opposed to
  statistical/ML models.

  The extracted scholarship requirements (percentage, income, etc.) are
  compared against student-provided values using simple comparators.

  Important:
  ----------
  This module can only check fields for which requirements were successfully
  extracted from the document. It does NOT fabricate requirements.
  It uses cautious wording and never officially declares eligibility.
"""

import re


# ---------------------------------------------------------------------------
# Helper: Parse Currency Amounts
# ---------------------------------------------------------------------------


def _parse_amount(text: str) -> float:
    """
    Extract the first numeric value from a currency string.

    Examples:
        "₹4,00,000" → 400000.0
        "Rs. 2,50,000 per annum" → 250000.0
        "INR 150000" → 150000.0

    Returns float or None if no number found.
    """
    if not text or text.strip().lower() in ("not found in the document.", ""):
        return None

    # Remove currency symbols and letters, keep digits and commas
    cleaned = re.sub(r"[₹$€£]|Rs\.?|INR", "", text, flags=re.IGNORECASE)
    # Extract first number with optional commas
    match = re.search(r"[\d,]+(?:\.\d+)?", cleaned)
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    return None


def _parse_percentage(text: str) -> float:
    """
    Extract a numeric percentage value from a string.

    Examples:
        "75%" → 75.0
        "60 percent" → 60.0
        "Not found in the document." → None
    """
    if not text or "not found" in text.lower():
        return None

    match = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:%|percent)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Eligibility Check Logic
# ---------------------------------------------------------------------------


def check_eligibility(student_info: dict, extracted_info: dict) -> dict:
    """
    Compare student-provided information against extracted scholarship requirements.

    Parameters
    ----------
    student_info : dict with keys:
        'percentage'  : str or float — student's marks/CGPA
        'income'      : str or float — annual family income
        'course'      : str — student's current course
        'age'         : str or int — student's age (optional)
        'category'    : str — student's category (optional)

    extracted_info : dict — output from information_extractor.extract_all()

    Returns
    -------
    dict with keys:
        'results'     : list of dicts, each with:
                          'field'    : str — requirement name
                          'required' : str — what the scholarship needs
                          'provided' : str — what the student provided
                          'status'   : 'met' | 'not_met' | 'unknown'
                          'message'  : str — human-readable result
        'overall'     : 'likely_eligible' | 'likely_ineligible' | 'insufficient_data'
        'summary'     : str — final overall message
    """
    results = []

    # -------------------------------------------------------------------
    # 1. Percentage Check
    # -------------------------------------------------------------------
    req_percentage_raw = extracted_info.get("percentage_requirement", "")
    req_percentage = _parse_percentage(req_percentage_raw)
    student_percentage_raw = str(student_info.get("percentage", "")).strip()

    if student_percentage_raw:
        try:
            student_pct = float(re.sub(r"[^\d.]", "", student_percentage_raw))
        except ValueError:
            student_pct = None

        if req_percentage is not None and student_pct is not None:
            met = student_pct >= req_percentage
            results.append({
                "field": "Minimum Percentage / Marks",
                "required": f"{req_percentage}%",
                "provided": f"{student_pct}%",
                "status": "met" if met else "not_met",
                "message": (
                    f"✅ Your score ({student_pct}%) meets the minimum requirement of {req_percentage}%."
                    if met else
                    f"❌ Your score ({student_pct}%) is below the minimum requirement of {req_percentage}%."
                ),
            })
        elif req_percentage is None:
            results.append({
                "field": "Minimum Percentage / Marks",
                "required": "Not specified in document",
                "provided": student_percentage_raw,
                "status": "unknown",
                "message": "ℹ️ Percentage requirement not found in the document. Cannot verify.",
            })
        elif student_pct is None:
            results.append({
                "field": "Minimum Percentage / Marks",
                "required": f"{req_percentage}%",
                "provided": "Not provided",
                "status": "unknown",
                "message": "ℹ️ Please enter a valid percentage to check this requirement.",
            })
    else:
        if req_percentage is not None:
            results.append({
                "field": "Minimum Percentage / Marks",
                "required": f"{req_percentage}%",
                "provided": "Not provided",
                "status": "unknown",
                "message": "ℹ️ Please enter your percentage to check this requirement.",
            })

    # -------------------------------------------------------------------
    # 2. Income Check
    # -------------------------------------------------------------------
    req_income_raw = extracted_info.get("income_requirement", "")
    req_income = _parse_amount(req_income_raw)
    student_income_raw = str(student_info.get("income", "")).strip()

    if student_income_raw:
        student_income = _parse_amount(student_income_raw)
        if student_income is None:
            try:
                student_income = float(re.sub(r"[^\d.]", "", student_income_raw))
            except ValueError:
                student_income = None

        if req_income is not None and student_income is not None:
            # For income, the requirement is usually "should NOT exceed" the limit
            met = student_income <= req_income
            req_formatted = f"₹{req_income:,.0f}"
            stu_formatted = f"₹{student_income:,.0f}"
            results.append({
                "field": "Annual Family Income",
                "required": f"Up to {req_formatted}",
                "provided": stu_formatted,
                "status": "met" if met else "not_met",
                "message": (
                    f"✅ Your family income ({stu_formatted}) is within the limit ({req_formatted})."
                    if met else
                    f"❌ Your family income ({stu_formatted}) exceeds the limit ({req_formatted})."
                ),
            })
        elif req_income is None:
            results.append({
                "field": "Annual Family Income",
                "required": "Not specified in document",
                "provided": student_income_raw,
                "status": "unknown",
                "message": "ℹ️ Income requirement not found in the document. Cannot verify.",
            })
    else:
        if req_income is not None:
            results.append({
                "field": "Annual Family Income",
                "required": f"₹{req_income:,.0f}",
                "provided": "Not provided",
                "status": "unknown",
                "message": "ℹ️ Please enter your annual family income to check this requirement.",
            })

    # -------------------------------------------------------------------
    # 3. Course Check (keyword matching)
    # -------------------------------------------------------------------
    student_course = student_info.get("course", "").strip()
    eligibility_text = extracted_info.get("eligibility", "").lower()

    if student_course:
        course_keywords = ["undergraduate", "ug", "btech", "b.tech", "be", "b.e",
                           "bsc", "b.sc", "ba", "b.a", "bcom", "b.com",
                           "postgraduate", "pg", "mtech", "m.tech", "msc", "m.sc",
                           "diploma", "engineering", "arts", "science", "commerce"]

        course_lower = student_course.lower()
        mentioned_in_doc = any(kw in eligibility_text for kw in course_keywords)

        if mentioned_in_doc:
            course_match = any(kw in eligibility_text for kw in course_lower.split())
            results.append({
                "field": "Course Eligibility",
                "required": "Specific course(s) mentioned in document",
                "provided": student_course,
                "status": "unknown",
                "message": (
                    "ℹ️ Please verify your course against the eligibility section — "
                    "specific course requirements are mentioned in the document."
                ),
            })
        else:
            results.append({
                "field": "Course Eligibility",
                "required": "Not specified in document",
                "provided": student_course,
                "status": "unknown",
                "message": "ℹ️ Course requirement not found in the document. Cannot verify.",
            })

    # -------------------------------------------------------------------
    # 4. Determine Overall Result
    # -------------------------------------------------------------------
    if not results:
        return {
            "results": [],
            "overall": "insufficient_data",
            "summary": (
                "⚠️ Not enough information could be extracted from the document "
                "to perform an eligibility check. Please enter your details and "
                "ensure the scholarship document contains eligibility requirements."
            ),
        }

    statuses = [r["status"] for r in results]
    met_count = statuses.count("met")
    not_met_count = statuses.count("not_met")
    unknown_count = statuses.count("unknown")

    if not_met_count > 0:
        overall = "likely_ineligible"
        summary = (
            "⚠️ Based on the information extracted from the document, "
            "you may NOT meet one or more eligibility criteria. "
            "Please review the details above carefully."
        )
    elif met_count > 0 and unknown_count == 0:
        overall = "likely_eligible"
        summary = (
            "✅ Based on the information extracted from the document, "
            "you appear to meet the available eligibility criteria. "
            "This is not an official eligibility determination — "
            "please verify directly with the scholarship provider."
        )
    else:
        overall = "insufficient_data"
        summary = (
            "ℹ️ Some eligibility criteria could not be verified due to "
            "missing information in the document or incomplete student details. "
            "Please check the eligibility section manually."
        )

    return {
        "results": results,
        "overall": overall,
        "summary": summary,
    }
