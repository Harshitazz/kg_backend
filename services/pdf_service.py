"""Compatibility layer for the PDF service.

The functional implementation now lives under the smaller module tree in
services/pdf/*. This file preserves the historical imports used throughout the
application so the refactor remains non-breaking.
"""

from kg_backend.utils.pdf.common import (
    AskPDFRequest,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SEARCH_K,
    EMBEDDING_MODEL_NAME,
    get_embedding_model,
    set_llm,
)
from kg_backend.utils.pdf.ingestion import ask_pdf_endpoint, process_pdf_documents, upload_pdfs_endpoint
from kg_backend.utils.pdf.ocr import extract_text_from_pdf, extract_text_with_ocr_fallback

__all__ = [
    "AskPDFRequest",
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_SEARCH_K",
    "EMBEDDING_MODEL_NAME",
    "get_embedding_model",
    "set_llm",
    "process_pdf_documents",
    "upload_pdfs_endpoint",
    "ask_pdf_endpoint",
    "extract_text_from_pdf",
    "extract_text_with_ocr_fallback",
]

