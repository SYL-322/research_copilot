"""Fetch metadata from the arXiv Atom API."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote

import httpx

from ingest.exceptions import ArxivError

logger = logging.getLogger(__name__)

# arXiv Atom namespace
_ATOM: Final = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS: Final = "{http://arxiv.org/schemas/atom}"

# e.g. 2401.12345v2, cs.AI/0001001
_ARXIV_ID_CORE: Final = re.compile(
    r"(?P<id>(?:[\w.-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)",
    re.IGNORECASE,
)

# URLs: /abs/, /pdf/, old-style
_ARXIV_URL: Final = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<id>[\w./-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArxivMetadata:
    """Metadata returned from the arXiv API."""

    arxiv_id: str
    """Canonical id including version suffix if present in the feed (e.g. 2401.12345v1)."""
    arxiv_id_base: str
    """Id without version (for stable deduplication)."""
    title: str
    authors: list[str]
    abstract: str
    published: date | None
    updated: datetime | None
    pdf_url: str
    abs_url: str
    primary_category: str | None


def normalize_arxiv_id(raw: str) -> str:
    """
    Strip whitespace and common prefixes (``arxiv:``, ``arXiv:``).

    Does not validate format.
    """
    s = raw.strip()
    for prefix in ("arxiv:", "arXiv:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :].strip()
    return s


def parse_arxiv_id_from_url_or_id(text: str) -> str | None:
    """
    Extract an arXiv identifier from a bare id or a URL.

    Returns the matched id string (may include version), or None.
    """
    s = text.strip()
    m = _ARXIV_URL.search(s)
    if m:
        raw_id = m.group("id")
        if raw_id.endswith(".pdf"):
            raw_id = raw_id[: -len(".pdf")]
        return normalize_arxiv_id(raw_id)
    s2 = normalize_arxiv_id(s)
    m2 = _ARXIV_ID_CORE.search(s2)
    if m2:
        return m2.group("id")
    return None


def arxiv_id_without_version(arxiv_id: str) -> str:
    """Strip a trailing ``vN`` version suffix if present."""
    return re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)


def fetch_arxiv_metadata(
    arxiv_id: str,
    *,
    timeout: float = 30.0,
) -> ArxivMetadata:
    """
    Fetch paper metadata from the arXiv Atom API.

    Parameters
    ----------
    arxiv_id
        Bare id or id extracted via :func:`parse_arxiv_id_from_url_or_id`.
    timeout
        HTTP timeout in seconds.

    Raises
    ------
    ArxivError
        If the id is invalid, the feed is empty, or the request fails.
    """
    aid = parse_arxiv_id_from_url_or_id(arxiv_id)
    if not aid:
        raise ArxivError(f"Could not parse arXiv id from: {arxiv_id!r}")

    # id_list must be URL-encoded
    url = f"http://export.arxiv.org/api/query?id_list={quote(aid, safe='')}"
    logger.info("Fetching arXiv metadata: %s", url)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.exception("arXiv HTTP error")
        raise ArxivError(f"arXiv request failed: {e}") from e

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        raise ArxivError(f"Invalid arXiv API response XML: {e}") from e

    entries = root.findall(f"{_ATOM}entry")
    if not entries:
        raise ArxivError(f"No arXiv entry found for id_list={aid!r}")

    entry = entries[0]
    title_el = entry.find(f"{_ATOM}title")
    title = _clean_text(title_el.text if title_el is not None else "") or "(untitled)"

    authors: list[str] = []
    for author in entry.findall(f"{_ATOM}author"):
        name_el = author.find(f"{_ATOM}name")
        if name_el is not None and name_el.text:
            authors.append(_clean_text(name_el.text))

    summary_el = entry.find(f"{_ATOM}summary")
    abstract = _clean_text(summary_el.text if summary_el is not None else "")

    published = _parse_date(entry.find(f"{_ATOM}published"))
    updated = _parse_datetime(entry.find(f"{_ATOM}updated"))

    # PDF link: usually rel="related" type="application/pdf"
    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("type") == "application/pdf" and link.get("href"):
            pdf_url = link.get("href", "").strip()
            break
    if not pdf_url:
        id_for_url = arxiv_id_without_version(aid)
        pdf_url = f"https://arxiv.org/pdf/{id_for_url}.pdf"

    abs_url = f"https://arxiv.org/abs/{arxiv_id_without_version(aid)}"

    prim = entry.find(f"{_ARXIV_NS}primary_category")
    primary_category = prim.get("term") if prim is not None else None

    return ArxivMetadata(
        arxiv_id=aid,
        arxiv_id_base=arxiv_id_without_version(aid),
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        abs_url=abs_url,
        primary_category=primary_category,
    )


def download_arxiv_pdf(
    pdf_url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
) -> Path:
    """
    Download a PDF from ``pdf_url`` to ``dest`` (parent dirs created).

    Returns
    -------
    Path
        Resolved destination path.
    """
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading PDF to %s", dest)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(pdf_url)
            r.raise_for_status()
            dest.write_bytes(r.content)
    except httpx.HTTPError as e:
        logger.exception("PDF download failed")
        raise ArxivError(f"Failed to download PDF: {e}") from e
    return dest


def _clean_text(s: str) -> str:
    return " ".join(s.split())


def _parse_date(el: ET.Element | None) -> date | None:
    if el is None or not el.text:
        return None
    raw = el.text.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _parse_datetime(el: ET.Element | None) -> datetime | None:
    if el is None or not el.text:
        return None
    raw = el.text.strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None