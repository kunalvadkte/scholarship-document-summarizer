"""
setup_nlp.py
============
One-time setup script to download all required NLP resources.

Run this ONCE before launching the application:
    python setup_nlp.py

This downloads:
  - NLTK: punkt_tab, punkt, stopwords, wordnet, omw-1.4
  - spaCy: en_core_web_sm (lightweight English model)
"""

import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("  Scholarship Document Summarizer - NLP Setup")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# Step 1: Download NLTK resources
# ---------------------------------------------------------------------------
print("📦 Downloading NLTK resources...")

import nltk

nltk_resources = [
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("punkt", "tokenizers/punkt"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
]

for resource_name, resource_path in nltk_resources:
    try:
        nltk.data.find(resource_path)
        print(f"  ✅ {resource_name} — already downloaded")
    except LookupError:
        print(f"  ⬇️  Downloading {resource_name}...")
        result = nltk.download(resource_name, quiet=False)
        if result:
            print(f"  ✅ {resource_name} — downloaded successfully")
        else:
            print(f"  ⚠️  {resource_name} — download may have failed. "
                  f"Try: python -c \"import nltk; nltk.download('{resource_name}')\"")

print()

# ---------------------------------------------------------------------------
# Step 2: Download spaCy model
# ---------------------------------------------------------------------------
print("📦 Downloading spaCy English model (en_core_web_sm)...")

try:
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("  ✅ en_core_web_sm — already installed")
    except OSError:
        print("  ⬇️  Installing en_core_web_sm...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  ✅ en_core_web_sm — installed successfully")
        else:
            print(f"  ❌ Failed to install en_core_web_sm.")
            print(f"     Please run manually: python -m spacy download en_core_web_sm")
            print(f"     Error: {result.stderr}")
except ImportError:
    print("  ❌ spaCy is not installed. Please run: pip install spacy")

print()
print("=" * 60)
print("✅ Setup complete! Run the application with:")
print("   streamlit run app.py")
print("=" * 60)
