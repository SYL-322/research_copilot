"""Backward-compatible exports for PDF extraction (see :mod:`ingest.pdf_loader`)."""

from __future__ import annotations

from pathlib import Path

from ingest.pdf_loader import extract_from_pdf


def extract_text_from_pdf(path: Path) -> str:
    """Return plain text from a PDF (delegates to :func:`ingest.pdf_loader.extract_from_pdf`)."""
    return extract_from_pdf(path).full_text
