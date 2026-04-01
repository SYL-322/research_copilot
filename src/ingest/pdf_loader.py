"""Extract text from research PDFs (pdfplumber primary, PyPDF fallback)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ingest.exceptions import PdfLoadError

logger = logging.getLogger(__name__)

# Typical section headings in ML/CV papers (case-insensitive line match).
_SECTION_KEYWORDS: tuple[str, ...] = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methodology",
    "approach",
    "model",
    "experiment",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "references",
    "acknowledgment",
    "appendix",
)

# Line looks like "1 Introduction" or "2. Related Work"
_NUMBERED_HEADING = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][\w\s\-]{2,80})\s*$",
)
# Single-line ALL CAPS heading (short)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-–]{3,80}$")


@dataclass
class SectionBlock:
    """A contiguous block of text with an optional inferred heading."""

    title: str | None
    text: str


@dataclass
class PdfExtractionResult:
    """Full document text plus coarse section splits."""

    full_text: str
    sections: list[SectionBlock] = field(default_factory=list)
    source: str = "pdfplumber"
    """Which backend produced ``full_text`` (``pdfplumber`` or ``pypdf``)."""


def extract_from_pdf(path: Path) -> PdfExtractionResult:
    """
    Extract plain text from a PDF on disk.

    Uses pdfplumber first (better for multi-column layouts); falls back to PyPDF
    if pdfplumber fails or returns empty text.

    Raises
    ------
    PdfLoadError
        If the file is missing, encrypted, or no text could be extracted.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PdfLoadError(f"PDF not found: {path}")

    text: str | None = None
    source = "pdfplumber"
    try:
        text = _extract_pdfplumber(path)
    except Exception as e:
        logger.warning("pdfplumber failed (%s); trying PyPDF", e)

    if not text or not text.strip():
        source = "pypdf"
        try:
            text = _extract_pypdf(path)
        except Exception as e:
            logger.exception("PyPDF extraction failed")
            raise PdfLoadError(f"Could not extract text from PDF: {e}") from e

    if not text or not text.strip():
        raise PdfLoadError(f"No extractable text in PDF: {path}")

    full_text = _normalize_whitespace(text)
    sections = infer_sections(full_text)
    return PdfExtractionResult(full_text=full_text, sections=sections, source=source)


def _extract_pdfplumber(path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def _extract_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as e:
        raise PdfLoadError(f"Invalid or encrypted PDF: {e}") from e
    if reader.is_encrypted:
        raise PdfLoadError("PDF is encrypted; cannot extract text without password")
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _normalize_whitespace(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 2:
                out.append("")
        else:
            blank = 0
            out.append(ln.strip())
    return "\n".join(out).strip()


def infer_sections(full_text: str) -> list[SectionBlock]:
    """
    Split ``full_text`` into coarse sections using heuristics.

    Falls back to a single block if no headings are detected.
    """
    lines = full_text.split("\n")
    blocks: list[SectionBlock] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append(SectionBlock(title=current_title, text=body))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue
        if _is_heading_line(stripped):
            flush()
            current_title = stripped
            continue
        current_lines.append(stripped)

    flush()

    if not blocks:
        return [SectionBlock(title=None, text=full_text.strip())]
    return blocks


def _is_heading_line(line: str) -> bool:
    if len(line) > 120:
        return False
    low = line.lower().strip("# ")
    for kw in _SECTION_KEYWORDS:
        if low == kw or low.startswith(kw + " ") or low.startswith(kw + ":"):
            return True
    if _NUMBERED_HEADING.match(line):
        return True
    if len(line) < 80 and _ALL_CAPS_HEADING.match(line) and len(line.split()) <= 8:
        return True
    return False
