"""Ingest PDFs, arXiv, and URLs into raw text and database rows."""

from ingest.arxiv_client import ArxivMetadata, fetch_arxiv_metadata, parse_arxiv_id_from_url_or_id
from ingest.chunker import TextChunk, chunk_sections
from ingest.exceptions import ArxivError, IngestError, PdfLoadError, UnsupportedInputError
from ingest.paper_ingestor import IngestResult, ingest, ingest_with_connection
from ingest.pdf_loader import PdfExtractionResult, SectionBlock, extract_from_pdf
from ingest.url_ingest import fetch_url_text

__all__ = [
    "ArxivError",
    "ArxivMetadata",
    "IngestError",
    "IngestResult",
    "PdfExtractionResult",
    "PdfLoadError",
    "SectionBlock",
    "TextChunk",
    "UnsupportedInputError",
    "chunk_sections",
    "extract_from_pdf",
    "fetch_arxiv_metadata",
    "fetch_url_text",
    "ingest",
    "ingest_with_connection",
    "parse_arxiv_id_from_url_or_id",
]
