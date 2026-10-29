# 🎓 Scholarship Document Summarizer

A complete **NLP Mini Project** for BTech students — upload a scholarship PDF and instantly extract key information, generate a concise summary, and optionally check your eligibility.

---

## 📋 Project Description

The **Scholarship Document Summarizer** is a locally running web application that processes scholarship PDF documents using Natural Language Processing techniques. It demonstrates a complete NLP pipeline — from raw PDF input to structured output — without requiring any paid API or external LLM.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Text Extraction | Extracts text from multi-page PDFs using PyMuPDF |
| 🧹 NLP Preprocessing | Tokenization, stopword removal, lemmatization via NLTK |
| 🔍 Information Extraction | Regex + spaCy NER to extract scholarship fields |
| 🧠 TF-IDF Summarization | Extractive summarization ranking key sentences |
| ✅ Eligibility Checker | Rule-based comparison of student details vs. requirements |
| 📊 Results Dashboard | Clean, structured display of all extracted information |
| ⚠️ Error Handling | Graceful handling of invalid, empty, or scanned PDFs |

---

## 🧪 NLP Techniques Used

| Technique | Library | Purpose |
|---|---|---|
| **Sentence Tokenization** | NLTK | Split document into sentences |
| **Word Tokenization** | NLTK | Split text into individual tokens |
| **Stopword Removal** | NLTK | Remove common low-meaning words |
| **Lemmatization** | NLTK (WordNet) | Reduce words to base form |
| **TF-IDF Vectorization** | scikit-learn | Score sentence importance |
| **Named Entity Recognition** | spaCy | Identify organizations, dates |
| **Regular Expressions** | Python `re` | Extract amounts, percentages, dates |
| **Keyword Matching** | Custom | Find eligibility/application sentences |
| **Extractive Summarization** | Custom (TF-IDF) | Select top-scoring sentences |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
PDF Validation (PyMuPDF)
    ↓
Text Extraction (page-by-page)
    ↓
NLP Preprocessing (NLTK)
  ├── Sentence Tokenization
  ├── Word Tokenization
  ├── Stopword Removal
  ├── Punctuation Removal
  └── Lemmatization
    ↓
Information Extraction
  ├── Scholarship Name (regex + NER)
  ├── Organization (spaCy NER + regex)
  ├── Amount (regex)
  ├── Percentage Requirement (regex)
  ├── Income Limit (regex)
  ├── Deadline (regex)
  ├── Eligibility (keyword sentences)
  ├── Required Documents (keyword matching)
  └── Application Process (keyword sentences)
    ↓
TF-IDF Extractive Summarization
  ├── Sentence Vectorization
  ├── Importance Scoring
  ├── Keyword Bonus Scoring
  ├── Sentence Ranking
  └── Order-Preserving Selection
    ↓
Results Dashboard (Streamlit)
    ↓
Eligibility Checker (Rule-based)
```

---

## 🔧 Technology Stack

- **Python 3.10+**
- **Streamlit** — Web interface
- **PyMuPDF (fitz)** — PDF text extraction
- **NLTK** — NLP preprocessing (tokenization, stopwords, lemmatization)
- **spaCy** — Named Entity Recognition
- **scikit-learn** — TF-IDF vectorization
- **Pandas** — Data handling
- **Python re** — Regular expression patterns

---

## 📁 Project Structure

```
scholarship-document-summarizer/
│
├── app.py                    ← Main Streamlit application
├── setup_nlp.py              ← One-time NLP resource downloader
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── modules/
│   ├── __init__.py
│   ├── pdf_extractor.py      ← PDF text extraction (PyMuPDF)
│   ├── preprocessing.py      ← NLP preprocessing pipeline (NLTK)
│   ├── information_extractor.py ← Regex + spaCy extraction
│   ├── summarizer.py         ← TF-IDF extractive summarization
│   └── eligibility_checker.py ← Rule-based eligibility check
│
├── data/
│   └── sample_scholarship.txt ← Sample/demo scholarship data
│
└── tests/
    ├── test_extractor.py     ← Tests for information extractor
    ├── test_summarizer.py    ← Tests for TF-IDF summarizer
    └── test_preprocessing.py ← Tests for NLP preprocessing
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone / Navigate to the project

```bash
cd scholarship-document-summarizer
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download NLP resources (run once)

```bash
python setup_nlp.py
```

This will:
- Download NLTK resources: `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`
- Download spaCy English model: `en_core_web_sm`

If spaCy model installation fails, run manually:
```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Run the application

```bash
streamlit run app.py
```

*Note for Windows users*: If `streamlit` is not recognized directly in PowerShell, use:
```bash
python -m streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🚀 How to Use

1. **Upload** a scholarship PDF document using the file uploader.
2. Click **"Analyse Scholarship"** to start the NLP pipeline.
3. View the **Results Dashboard**:
   - Scholarship Overview (name, provider, amount, deadline)
   - Eligibility Criteria (percentage, income)
   - Required Documents (auto-detected checklist)
   - Application Process
   - NLP-Generated TF-IDF Summary
4. Scroll down and enter your details in the **Eligibility Checker** to compare against extracted requirements.

---

## 🧪 Running Tests

```bash
# Run all tests
python tests/test_preprocessing.py
python tests/test_summarizer.py
python tests/test_extractor.py

# Or with pytest (if installed)
pip install pytest
pytest tests/ -v
```

---

## 📄 Sample Output

Given a scholarship document, the application extracts:

```
Scholarship Name:        ABC Education Foundation Scholarship 2026
Organization:            ABC Education Foundation
Scholarship Amount:      Rs. 50,000 per year
Application Deadline:    30 September 2026

Minimum Percentage:      75%
Income Requirement:      Annual family income must not exceed Rs. 4,00,000

Required Documents:
  ✅ Aadhaar Card
  ✅ Marksheet / Mark Sheet
  ✅ Income Certificate
  ✅ Bonafide Certificate
  ✅ Bank Account Details
  ✅ Passport Photograph
  ✅ Domicile Certificate
  ✅ College / Student ID Card

NLP Summary:
  "The ABC Education Foundation Scholarship 2026 is a merit-cum-means scholarship
  offered to undergraduate students from economically weaker sections. Applicants
  must have obtained a minimum of 75% marks. The annual family income must not
  exceed Rs. 4,00,000 per annum. Selected students will receive Rs. 50,000 per year."
```

---

## ⚠️ Limitations

- **Scanned PDFs**: The application cannot extract text from image-based scanned PDFs (OCR is not included).
- **Complex layouts**: Multi-column PDFs or PDFs with complex tables may have extraction issues.
- **Extraction accuracy**: The extraction accuracy depends on the quality and structure of the input document.
- **Language**: Only English documents are fully supported.
- **No fabrication**: If a field cannot be found, the app shows "Not found in the document." and never generates fake values.

---

## 🔮 Future Enhancements

- [ ] OCR support for scanned PDFs (using `pytesseract`)
- [ ] Multi-language support
- [ ] Abstractive summarization using transformer models (BART, T5)
- [ ] Scholarship database to compare multiple scholarships
- [ ] Email notification for approaching deadlines
- [ ] Export results to PDF/Excel report

---

## 📚 Academic Explanation (for Viva)

### Why TF-IDF for Summarization?
TF-IDF (Term Frequency - Inverse Document Frequency) identifies words that are important in a sentence but not common across the whole document. Sentences containing such words are likely to be more informative — making TF-IDF ideal for extractive summarization without requiring any ML model training.

### Why Extractive, not Abstractive?
Extractive summarization selects real sentences from the document, which is:
- Interpretable: can always trace back to the source
- Reliable: never introduces hallucinated information
- Lightweight: no large model required

### Why Regex for Information Extraction?
Scholarship documents follow structured patterns (dates, currency amounts, percentages). Regular expressions are efficient, transparent, and explainable — ideal for rule-based information extraction tasks.

---

## 📜 Disclaimer

This tool is for educational/academic purposes only.
It is not an official eligibility determination system.
Always verify scholarship information directly with the official provider.

---

*NLP Mini Project — BTech*
