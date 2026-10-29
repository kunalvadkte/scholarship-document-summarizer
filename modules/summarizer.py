"""
summarizer.py
=============
Extractive Summarization using TF-IDF (Term Frequency - Inverse Document Frequency).

NLP Concepts Covered:
----------------------
TF-IDF (Term Frequency - Inverse Document Frequency):
  - TF  (Term Frequency): How often a word appears in a sentence.
         TF(word, sentence) = count(word in sentence) / total words in sentence
  - IDF (Inverse Doc Freq): How rare a word is across all sentences.
         IDF(word) = log(total sentences / sentences containing word)
  - TF-IDF = TF × IDF → gives high scores to words that are frequent in a
             sentence but rare overall (i.e., important/distinctive words).

Extractive Summarization:
  - We do NOT generate new text (that would be abstractive summarization).
  - We SCORE existing sentences and SELECT the most important ones.
  - This is interpretable, requires no large model, and works well for
    structured documents like scholarship forms/notices.

Process:
  1. Split document into sentences
  2. Vectorise sentences using TF-IDF (scikit-learn's TfidfVectorizer)
  3. Sum each sentence's TF-IDF scores to get an importance score
  4. Add a domain-specific bonus for sentences with scholarship keywords
  5. Rank sentences by score
  6. Select top N sentences
  7. Restore original order (so the summary reads naturally)
  8. Return summary
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------------------------------------------------------------------
# Domain Keywords
# ---------------------------------------------------------------------------
# These keywords give an extra score boost to sentences that contain
# scholarship-relevant information. This keeps the TF-IDF generic while
# biasing selection toward the most useful sentences for our use case.

SCHOLARSHIP_KEYWORDS = {
    "scholarship", "eligibility", "eligible", "amount", "deadline",
    "income", "marks", "percentage", "documents", "apply", "application",
    "selection", "benefit", "award", "stipend", "grant", "merit",
    "criteria", "requirement", "last date", "submit", "academic",
    "annual", "family", "certificate", "minimum", "maximum",
}

KEYWORD_BONUS = 0.15  # Additional weight per matching keyword found


# ---------------------------------------------------------------------------
# Main Summarizer Function
# ---------------------------------------------------------------------------


def summarize(text: str, num_sentences: int = None, min_sentences: int = 3,
              max_sentences: int = 7) -> dict:
    """
    Generate an extractive summary using TF-IDF sentence scoring.

    Parameters
    ----------
    text          : str — full document text
    num_sentences : int or None — if set, force this many sentences in summary
    min_sentences : int — minimum summary length (default 3)
    max_sentences : int — maximum summary length (default 7)

    Returns
    -------
    dict with keys:
        'success'        : bool
        'summary'        : str — the generated summary
        'sentence_count' : int — number of sentences in the summary
        'scored_sentences': list of (sentence, score) tuples (for debugging)
        'error'          : str — error message if success is False
    """
    if not text or not text.strip():
        return {
            "success": False,
            "summary": "",
            "sentence_count": 0,
            "scored_sentences": [],
            "error": "No text provided for summarization.",
        }

    # -----------------------------------------------------------------------
    # Step 1: Sentence Splitting
    # -----------------------------------------------------------------------
    sentences = _split_into_sentences(text)

    if len(sentences) == 0:
        return {
            "success": False,
            "summary": "",
            "sentence_count": 0,
            "scored_sentences": [],
            "error": "Could not extract sentences from the document.",
        }

    # If the document is very short, return the full text as-is
    if len(sentences) <= min_sentences:
        return {
            "success": True,
            "summary": " ".join(sentences),
            "sentence_count": len(sentences),
            "scored_sentences": [(s, 1.0) for s in sentences],
            "error": "",
        }

    # -----------------------------------------------------------------------
    # Step 2: TF-IDF Vectorisation
    # -----------------------------------------------------------------------
    # TfidfVectorizer converts each sentence into a vector of TF-IDF scores.
    # Each dimension corresponds to one vocabulary word.
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",       # Remove common English stopwords
            ngram_range=(1, 2),         # Consider single words and bigrams
            min_df=1,                   # Include words that appear at least once
            max_df=0.9,                 # Ignore words in >90% of sentences
            sublinear_tf=True,          # Use log(1+tf) to dampen high frequencies
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Fallback if TF-IDF fails (e.g. all sentences are stopwords only)
        return {
            "success": True,
            "summary": " ".join(sentences[:max_sentences]),
            "sentence_count": min(len(sentences), max_sentences),
            "scored_sentences": [(s, 0.0) for s in sentences],
            "error": "",
        }

    # -----------------------------------------------------------------------
    # Step 3: Calculate Sentence Importance Scores
    # -----------------------------------------------------------------------
    # Sum the TF-IDF values across all features for each sentence.
    # A sentence with many high-scoring terms gets a high total score.
    tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # -----------------------------------------------------------------------
    # Step 4: Domain Keyword Bonus
    # -----------------------------------------------------------------------
    # Give extra weight to sentences that contain scholarship-specific terms.
    # This is a lightweight heuristic on top of the generic TF-IDF score.
    keyword_bonuses = np.zeros(len(sentences))
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        for keyword in SCHOLARSHIP_KEYWORDS:
            if keyword in sentence_lower:
                keyword_bonuses[i] += KEYWORD_BONUS

    # Normalise keyword bonuses relative to the TF-IDF scores
    if tfidf_scores.max() > 0:
        combined_scores = tfidf_scores / tfidf_scores.max() + keyword_bonuses
    else:
        combined_scores = keyword_bonuses

    # -----------------------------------------------------------------------
    # Step 5: Determine Summary Length
    # -----------------------------------------------------------------------
    if num_sentences is not None:
        n = max(1, min(num_sentences, len(sentences)))
    else:
        # Scale summary length with document size
        doc_length = len(sentences)
        if doc_length <= 10:
            n = min_sentences
        elif doc_length <= 30:
            n = min(5, max_sentences)
        else:
            n = max_sentences

    # -----------------------------------------------------------------------
    # Step 6: Rank and Select Top Sentences
    # -----------------------------------------------------------------------
    # argsort returns indices that would sort the array in ascending order.
    # We take the last n indices (highest scores).
    ranked_indices = np.argsort(combined_scores)[-n:]

    # -----------------------------------------------------------------------
    # Step 7: Preserve Original Order
    # -----------------------------------------------------------------------
    # Sort selected sentence indices in their original document order
    # so the summary reads naturally (not in importance order).
    selected_indices = sorted(ranked_indices)

    # -----------------------------------------------------------------------
    # Step 8: Build Summary
    # -----------------------------------------------------------------------
    summary_sentences = [sentences[i] for i in selected_indices]
    summary = " ".join(summary_sentences)

    # Build scored list for optional display/debugging
    scored_sentences = sorted(
        zip(sentences, combined_scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "success": True,
        "summary": summary,
        "sentence_count": len(summary_sentences),
        "scored_sentences": scored_sentences,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Helper: Sentence Splitter
# ---------------------------------------------------------------------------


def _split_into_sentences(text: str) -> list:
    """
    Split text into sentences using a combination of NLTK and regex.
    Falls back gracefully if NLTK resources are unavailable.

    Parameters
    ----------
    text : str

    Returns
    -------
    list of str — individual sentences, filtered to minimum meaningful length
    """
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except LookupError:
        # Regex fallback: split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out very short sentences (less than 4 words) to avoid noise
    MIN_WORDS = 4
    sentences = [s.strip() for s in sentences if len(s.split()) >= MIN_WORDS]

    return sentences
