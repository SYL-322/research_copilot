"""Fetch content from HTTP(S) URLs."""

from __future__ import annotations


def fetch_url_text(url: str, *, timeout: float = 30.0) -> str:
    """
    Fetch a URL and return extractable text (HTML stripped or PDF via ingest).

    TODO: respect robots.txt policy for production; handle PDF links.
    """
    raise NotImplementedError
