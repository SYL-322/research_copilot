"""Ingestion-specific errors."""


class IngestError(Exception):
    """Base class for paper ingestion failures."""


class ArxivError(IngestError):
    """arXiv ID parsing or API failure."""


class PdfLoadError(IngestError):
    """PDF could not be read or decrypted."""


class UnsupportedInputError(IngestError):
    """Input string is not a supported URL or path."""
