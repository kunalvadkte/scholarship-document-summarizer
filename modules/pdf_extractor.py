"""
pdf_extractor.py
================
Handles PDF text extraction using PyMuPDF (fitz).

NLP Concept: This is the first step — data acquisition.
Before any NLP processing can occur, we must convert
the binary PDF into raw plain text.
"""

import fitz  # PyMuPDF


def extract_text(pdf_file) -> dict:
    """
    Extract text from an uploaded PDF file object.

    Parameters
    ----------
    pdf_file : file-like object (e.g. from Streamlit's file_uploader)

    Returns
    -------
    dict with keys:
        'success'  : bool
        'text'     : str — full extracted text (empty string on failure)
        'pages'    : int — number of pages processed
        'error'    : str — error message if success is False
    """
    try:
        # Read the bytes from the uploaded file object
        pdf_bytes = pdf_file.read()

        # Open the PDF from bytes using PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(doc) == 0:
            return {
                "success": False,
                "text": "",
                "pages": 0,
                "error": "The uploaded PDF has no pages.",
            }

        all_text = []

        # Iterate through each page and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")  # Extract plain text

            # Only add pages that have meaningful content
            if page_text and page_text.strip():
                all_text.append(page_text.strip())

        doc.close()

        combined_text = "\n\n".join(all_text)

        # Check if we actually got any text (scanned PDFs have no text layer)
        if not combined_text.strip():
            return {
                "success": False,
                "text": "",
                "pages": len(doc),
                "error": (
                    "No text could be extracted from this PDF. "
                    "It may be a scanned image-based PDF. "
                    "Please use a PDF with selectable text."
                ),
            }

        return {
            "success": True,
            "text": combined_text,
            "pages": len(all_text),
            "error": "",
        }

    except fitz.FileDataError:
        return {
            "success": False,
            "text": "",
            "pages": 0,
            "error": "The file could not be read as a valid PDF. Please upload a proper PDF file.",
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "pages": 0,
            "error": f"An unexpected error occurred during PDF extraction: {str(e)}",
        }


def extract_text_from_path(pdf_path: str) -> dict:
    """
    Extract text from a PDF given its file path.
    Useful for testing without a Streamlit file object.

    Parameters
    ----------
    pdf_path : str — absolute or relative path to the PDF

    Returns
    -------
    Same dict format as extract_text()
    """
    try:
        doc = fitz.open(pdf_path)

        if len(doc) == 0:
            return {
                "success": False,
                "text": "",
                "pages": 0,
                "error": "The PDF has no pages.",
            }

        all_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                all_text.append(page_text.strip())

        doc.close()

        combined_text = "\n\n".join(all_text)

        if not combined_text.strip():
            return {
                "success": False,
                "text": "",
                "pages": 0,
                "error": "No text found in PDF (possibly scanned).",
            }

        return {
            "success": True,
            "text": combined_text,
            "pages": len(all_text),
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "pages": 0,
            "error": str(e),
        }
