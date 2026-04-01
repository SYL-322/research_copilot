"""Backward-compatible arXiv helpers (see :mod:`ingest.arxiv_client`)."""

from __future__ import annotations

from ingest.arxiv_client import (
    ArxivMetadata,
    fetch_arxiv_metadata,
    parse_arxiv_id_from_url_or_id,
)

__all__ = ["ArxivMetadata", "fetch_arxiv_metadata", "parse_arxiv_id_from_url_or_id"]
