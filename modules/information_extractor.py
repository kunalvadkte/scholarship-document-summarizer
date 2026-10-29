r"""
information_extractor.py
========================
Extracts structured scholarship information from raw text using:
  - Regular Expressions (Regex): Pattern matching for amounts, dates, percentages
  - Keyword-based Extraction: Finding relevant sentences around known terms
  - spaCy Named Entity Recognition (NER): Identifying organisations and persons

NLP Concepts:
  - Regex: A formal language for describing text patterns.
           e.g., r"₹[\d,]+" matches "₹50,000" or "₹4,00,000"
  - NER: A subtask of information extraction that locates and classifies
         named entities (organisations, dates, money) in text.
  - Keyword Matching: Find sentences near domain-specific anchor words.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# spaCy Setup — graceful fallback if model is missing
# ---------------------------------------------------------------------------

_NLP = None  # spaCy language model (loaded lazily)


def _get_nlp():
    """Load spaCy model on first use. Returns None if unavailable."""
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
    except ImportError:
        _NLP = None
    return _NLP


# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

# Indian currency: ₹50,000 | Rs. 50,000 | INR 50,000 | 50000 per year
_AMOUNT_PATTERNS = [
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:per\s+(?:year|annum|month)|annually|p\.?a\.?))?",
    r"[\d,]+\s*(?:per\s+(?:year|annum|month)|annually)",
]

# Percentage: 75% | 75 percent | minimum 60% | at least 70 percent
_PERCENTAGE_PATTERNS = [
    r"(?:minimum|at\s+least|above|more\s+than|not\s+less\s+than|minimum\s+of)?\s*"
    r"(\d{2}(?:\.\d+)?)\s*(?:%|percent(?:age)?)",
    r"(\d{2}(?:\.\d+)?)\s*(?:%|percent(?:age)?)\s*(?:marks?|cgpa|score)?",
]

# Date patterns: 30 September 2026 | 30/09/2026 | 30-09-2026 | September 30, 2026
_DATE_PATTERNS = [
    r"\b(?:0?[1-9]|[12]\d|3[01])\s+(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+\d{4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{4}\b",
    r"\b(?:0?[1-9]|[12]\d|3[01])[-/](?:0?[1-9]|1[0-2])[-/]\d{4}\b",
    r"\b\d{4}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b",
]

# Income: annual family income below ₹4,00,000 | income should not exceed Rs. 5,00,000
_INCOME_PATTERNS = [
    r"(?:annual|family|household)?\s*income\s*(?:should\s+)?(?:not\s+)?(?:exceed|below|less\s+than|"
    r"more\s+than|above|up\s+to|limit(?:ed)?(?:\s+to)?)?\s*(?:₹|Rs\.?|INR)?\s*[\d,]+(?:\.\d+)?",
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?\s*(?:per\s+annum|annually|p\.?a\.?)?.*?income",
]

# ---------------------------------------------------------------------------
# Known Document Types (for document checklist)
# ---------------------------------------------------------------------------

KNOWN_DOCUMENTS = [
    "aadhaar card", "aadhar card", "aadhaar", "aadhar",
    "marksheet", "mark sheet", "marks sheet",
    "income certificate",
    "caste certificate", "caste",
    "bonafide certificate", "bonafide",
    "bank account", "bank passbook", "bank details",
    "passport photograph", "photograph", "photo",
    "domicile certificate", "domicile",
    "disability certificate", "disability",
    "birth certificate",
    "residence certificate", "residence proof",
    "fee receipt",
    "college id", "student id", "id card",
    "community certificate",
    "migration certificate",
    "transfer certificate",
]

# ---------------------------------------------------------------------------
# Eligibility Keywords
# ---------------------------------------------------------------------------

ELIGIBILITY_KEYWORDS = [
    "eligible", "eligibility", "criteria", "qualification", "qualify",
    "students", "percentage", "income", "age", "course", "category",
    "applicant", "candidate", "minimum", "maximum", "should", "must",
    "require", "requirement", "condition", "class", "pursuing",
    "studying", "enrolled", "admission",
]

# ---------------------------------------------------------------------------
# Application Process Keywords
# ---------------------------------------------------------------------------

APPLICATION_KEYWORDS = [
    "apply", "application", "how to apply", "registration", "register",
    "submit", "portal", "online", "website", "form", "procedure",
    "process", "step", "visit", "login", "upload",
]


# ===========================================================================
# Public API
# ===========================================================================


def extract_all(text: str) -> dict:
    """
    Master extraction function — runs all extractors and returns a dict
    containing all scholarship fields.

    Parameters
    ----------
    text : str — raw PDF text

    Returns
    -------
    dict with keys:
        scholarship_name, organization, amount, percentage_requirement,
        income_requirement, deadline, eligibility, documents, application_process
    """
    return {
        "scholarship_name": extract_scholarship_name(text),
        "organization": extract_organization(text),
        "amount": extract_amount(text),
        "percentage_requirement": extract_percentage(text),
        "income_requirement": extract_income(text),
        "deadline": extract_deadline(text),
        "eligibility": extract_eligibility(text),
        "documents": extract_documents(text),
        "application_process": extract_application_process(text),
    }


# ---------------------------------------------------------------------------
# Individual Extractors
# ---------------------------------------------------------------------------


def extract_scholarship_name(text: str) -> str:
    """
    Extract the scholarship name by looking for heading-like phrases.

    Strategy:
    1. Look for lines following headings: "Scholarship Name:", "Name:", etc.
    2. Look for lines containing "Scholarship" that appear at the top of the doc.
    3. Use spaCy NER as a fallback (looks for ORG/WORK_OF_ART entities).
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Pattern 1: Explicit label
    label_pattern = re.compile(
        r"(?:scholarship\s+name|name\s+of\s+scholarship|scheme\s+name|"
        r"name\s+of\s+scheme)\s*[:\-–]\s*(.+)",
        re.IGNORECASE
    )
    for line in lines:
        m = label_pattern.search(line)
        if m:
            return m.group(1).strip()

    # Pattern 2: Look at first 10 lines for a line containing "Scholarship"
    for line in lines[:10]:
        if "scholarship" in line.lower() and len(line.split()) <= 12:
            return line.strip()

    # Pattern 3: Line ending with "Scholarship" or "Fellowship" or "Award"
    title_pattern = re.compile(
        r"^(.{5,80}(?:scholarship|fellowship|award|grant|bursary))$",
        re.IGNORECASE | re.MULTILINE
    )
    m = title_pattern.search(text[:500])  # Only look in first 500 chars
    if m:
        return m.group(1).strip()

    # Pattern 4: spaCy NER fallback
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "WORK_OF_ART", "PRODUCT"):
                if "scholarship" in ent.text.lower() or "fellowship" in ent.text.lower():
                    return ent.text.strip()

    return NOT_FOUND


def extract_organization(text: str) -> str:
    """
    Extract the scholarship provider / organization name.

    Strategy:
    1. Look for explicit labels: "Organization:", "Provider:", "By:", "Offered by:"
    2. Use spaCy NER to find ORG entities near scholarship context.
    3. Keyword heuristic: lines containing "Foundation", "Trust", "Ministry", etc.
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Pattern 1: Explicit label
    label_pattern = re.compile(
        r"(?:organization|organisation|offered\s+by|provided\s+by|"
        r"sponsored\s+by|institution|trust|foundation|authority|"
        r"department|ministry|board|corporation|society|provider|issuer)\s*[:\-–]\s*(.+)",
        re.IGNORECASE
    )
    for line in lines:
        m = label_pattern.search(line)
        if m:
            return m.group(1).strip()

    # Pattern 2: spaCy NER — find ORG entities
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:2000])
        org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if org_entities:
            # Return the first/most prominent ORG entity
            return org_entities[0].strip()

    # Pattern 3: Keyword heuristic — lines containing org-related words
    org_keywords = re.compile(
        r"\b(foundation|trust|ministry|department|board|corporation|"
        r"government|institute|university|college|society|authority|"
        r"welfare|ngo|charitable)\b",
        re.IGNORECASE
    )
    for line in lines[:20]:
        if org_keywords.search(line) and len(line.split()) <= 10:
            return line.strip()

    return NOT_FOUND


def extract_amount(text: str) -> str:
    """
    Extract scholarship monetary amount using regex.

    Detects:
        ₹50,000 | Rs. 50,000 | INR 50,000 | 50000 per year
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    all_matches = []
    for pattern in _AMOUNT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        all_matches.extend(matches)

    if not all_matches:
        return NOT_FOUND

    # Prefer the first match that includes a currency symbol
    for match in all_matches:
        if any(sym in match for sym in ["₹", "Rs", "INR"]):
            return match.strip()

    return all_matches[0].strip()


def extract_percentage(text: str) -> str:
    """
    Extract minimum percentage requirement using regex.

    Detects:
        75% | minimum 60% | at least 70 percent | minimum percentage of 65%
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    # Look in sentences that are about percentage/marks requirements
    relevant_sentences = []
    for sent in text.split("."):
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in ["percent", "%", "marks", "cgpa", "score", "grade"]):
            relevant_sentences.append(sent)

    search_text = ". ".join(relevant_sentences) if relevant_sentences else text

    # Try each pattern
    for pattern in _PERCENTAGE_PATTERNS:
        matches = re.findall(pattern, search_text, re.IGNORECASE)
        if matches:
            # Return the first reasonable percentage (between 40 and 100)
            for match in matches:
                try:
                    val = float(match)
                    if 40 <= val <= 100:
                        return f"{match}%"
                except (ValueError, TypeError):
                    continue

    return NOT_FOUND


def extract_income(text: str) -> str:
    """
    Extract family income limit using regex.

    Detects:
        annual family income below ₹4,00,000
        income should not exceed Rs. 5,00,000
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    # Search in sentences mentioning income
    income_sentences = []
    for sent in re.split(r"[.\n]", text):
        if "income" in sent.lower():
            income_sentences.append(sent.strip())

    if not income_sentences:
        return NOT_FOUND

    for sent in income_sentences:
        # Find currency amounts near the word "income"
        amount_match = re.search(
            r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?", sent, re.IGNORECASE
        )
        if amount_match:
            # Return context around the match
            return sent.strip()[:200]

    # Try raw income patterns on full text
    for pattern in _INCOME_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()

    # Return the first income-related sentence
    if income_sentences:
        return income_sentences[0][:200]

    return NOT_FOUND


def extract_deadline(text: str) -> str:
    """
    Extract application deadline / last date using regex.

    Detects:
        30 September 2026 | 30/09/2026 | September 30, 2026
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    # Prefer sentences that mention deadline context
    deadline_sentences = []
    for sent in re.split(r"[.\n]", text):
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in ["last date", "deadline", "closing date",
                                            "due date", "apply by", "submit by",
                                            "final date", "application date"]):
            deadline_sentences.append(sent)

    search_texts = deadline_sentences if deadline_sentences else [text]

    for search_text in search_texts:
        for pattern in _DATE_PATTERNS:
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()

    # If no deadline context found, search full text for any date
    for pattern in _DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return date only if a deadline-related word is nearby
            for match in matches:
                idx = text.find(match)
                context = text[max(0, idx - 100):idx + 100].lower()
                if any(kw in context for kw in ["last date", "deadline", "apply",
                                                  "submit", "closing", "final"]):
                    return match.strip()

    return NOT_FOUND


def extract_eligibility(text: str) -> str:
    """
    Extract eligibility criteria by collecting relevant sentences/paragraphs.

    Finds sentences containing eligibility-related keywords and returns them
    as a structured block of text.
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

    relevant = []
    seen = set()

    for sent in sentences:
        sent_lower = sent.lower()
        # Count how many eligibility keywords appear in this sentence
        keyword_hits = sum(1 for kw in ELIGIBILITY_KEYWORDS if kw in sent_lower)

        if keyword_hits >= 1 and len(sent.split()) >= 5:
            # Deduplicate by comparing simplified versions
            simplified = re.sub(r"\s+", " ", sent.lower().strip())
            if simplified not in seen:
                seen.add(simplified)
                relevant.append(sent.strip())

    if not relevant:
        return NOT_FOUND

    # Limit to 10 most relevant sentences to keep it concise
    return "\n• ".join([""] + relevant[:10]).strip("• \n")


def extract_documents(text: str) -> list:
    """
    Extract required documents by searching for known document type names.

    Returns a list of document names found in the text.
    Only reports documents that are actually mentioned — does not assume.
    """
    if not text:
        return []

    text_lower = text.lower()
    found_documents = []
    seen_canonical = set()

    # Map display name → search terms
    document_map = {
        "Aadhaar Card": ["aadhaar card", "aadhar card", "aadhaar", "aadhar"],
        "Marksheet / Mark Sheet": ["marksheet", "mark sheet", "marks sheet", "grade sheet"],
        "Income Certificate": ["income certificate"],
        "Caste Certificate": ["caste certificate", "caste/community certificate"],
        "Bonafide Certificate": ["bonafide certificate", "bona fide certificate", "bonafide"],
        "Bank Account Details": ["bank account", "bank passbook", "cancelled cheque", "bank details"],
        "Passport Photograph": ["passport photograph", "passport size photograph", "recent photograph"],
        "Domicile Certificate": ["domicile certificate", "domicile"],
        "Disability Certificate": ["disability certificate"],
        "Birth Certificate": ["birth certificate"],
        "Residence Certificate": ["residence certificate", "residence proof", "address proof"],
        "Fee Receipt": ["fee receipt", "tuition fee receipt"],
        "College / Student ID Card": ["college id", "student id", "id card", "identity card"],
        "Community Certificate": ["community certificate"],
        "Migration Certificate": ["migration certificate"],
        "Transfer Certificate": ["transfer certificate"],
    }

    for display_name, search_terms in document_map.items():
        for term in search_terms:
            if term in text_lower:
                canonical = display_name
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    found_documents.append(display_name)
                break  # Found this document, move to next

    return found_documents


def extract_application_process(text: str) -> str:
    """
    Extract application instructions by finding relevant sentences/paragraphs.

    Looks for sections describing how to apply, registration process,
    submission instructions, and portal information.
    """
    NOT_FOUND = "Not found in the document."

    if not text:
        return NOT_FOUND

    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

    relevant = []
    seen = set()

    for sent in sentences:
        sent_lower = sent.lower()
        keyword_hits = sum(1 for kw in APPLICATION_KEYWORDS if kw in sent_lower)

        if keyword_hits >= 1 and len(sent.split()) >= 4:
            simplified = re.sub(r"\s+", " ", sent.lower().strip())
            if simplified not in seen:
                seen.add(simplified)
                relevant.append(sent.strip())

    if not relevant:
        return NOT_FOUND

    return "\n• ".join([""] + relevant[:8]).strip("• \n")
