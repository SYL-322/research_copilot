"""Backward-compatible topic search (see :mod:`topic.literature_search`)."""

from __future__ import annotations

from core.config import load_settings
from topic.literature_search import CandidatePaper, search_literature

# Legacy name used in early scaffold
PaperHit = CandidatePaper


def search_topic(query: str, *, max_results: int = 20) -> list[CandidatePaper]:
    """Search arXiv + Semantic Scholar and return deduplicated candidates."""
    return search_literature(query, max_results=max_results, settings=load_settings())
