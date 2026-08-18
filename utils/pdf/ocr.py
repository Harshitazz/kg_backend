import logging
from typing import Any

import fitz

try:
    import pytesseract
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract/Pillow not available, OCR fallback disabled")

logger = logging.getLogger(__name__)


def extract_text_with_ocr_fallback(page: fitz.Page, pdf_key: str, page_num: int) -> str:
    """Extract text from PDF page with OCR fallback if no text found."""
    text = page.get_text("text")

    if not text.strip():
        if not OCR_AVAILABLE:
            return ""

        try:
            try:
                pytesseract.get_tesseract_version()
            except Exception as exc:
                raise Exception(f"Tesseract OCR is not installed or not in PATH: {exc}. Please install Tesseract OCR.")

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")

            from io import BytesIO

            img = Image.open(BytesIO(img_data))
            text = pytesseract.image_to_string(img)
            if text.strip():
                print(f"Used OCR for page {page_num + 1} in {pdf_key}")
            else:
                print(f"OCR found no text on page {page_num + 1} in {pdf_key} (may be blank or poor quality)")
        except Exception as exc:
            error_msg = str(exc)
            if "Tesseract" in error_msg or "not installed" in error_msg.lower() or "PATH" in error_msg:
                raise Exception(f"OCR configuration error: {error_msg}")
            raise Exception(f"OCR processing failed for page {page_num + 1}: {error_msg}")

    return text


def extract_text_from_pdf(pdf_path: str):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text
