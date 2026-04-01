"""Search arXiv and Semantic Scholar for topic-relevant papers."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlencode

import httpx

from core.config import Settings

logger = logging.getLogger(__name__)

_ATOM: Final = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS: Final = "{http://arxiv.org/schemas/atom}"

ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_HEADERS: Final = {
    "User-Agent": "research_copilot/1.0 (+https://github.com/)",
}
SEARCH_RETRY_DELAYS: Final[tuple[float, ...]] = (1.0, 2.0)
_last_search_rate_limited_sources: set[str] = set()


class LiteratureSearchRateLimitError(RuntimeError):
    """Raised when literature search providers are temporarily rate-limiting us."""


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return max(0.0, min(seconds, 30.0))


def _mark_rate_limited(source: str) -> None:
    _last_search_rate_limited_sources.add(source)


def _should_retry_http_error(exc: httpx.HTTPStatusError) -> bool:
    code = exc.response.status_code
    return code == 429 or 500 <= code < 600


def _request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    attempts = len(SEARCH_RETRY_DELAYS) + 1
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    for attempt in range(attempts):
        try:
            response = client.request(method, url, params=params, headers=merged_headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if attempt >= attempts - 1 or not _should_retry_http_error(exc):
                raise
            retry_after = _parse_retry_after(exc.response.headers.get("Retry-After"))
            delay = retry_after if retry_after is not None else SEARCH_RETRY_DELAYS[attempt]
            logger.warning(
                "HTTP %s from %s; retrying in %.1fs (attempt %s/%s)",
                exc.response.status_code,
                url,
                delay,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)
        except httpx.TransportError:
            if attempt >= attempts - 1:
                raise
            delay = SEARCH_RETRY_DELAYS[attempt]
            logger.warning(
                "Transport error from %s; retrying in %.1fs (attempt %s/%s)",
                url,
                delay,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


@dataclass
class CandidatePaper:
    """Normalized metadata for a paper returned by literature search."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    venue: str | None = None
    source: str = ""
    published_iso: str | None = None
    """ISO-like publication timestamp when available (arXiv ``published``, SS ``publicationDate``)."""

    def as_prompt_dict(self) -> dict[str, object]:
        """Compact dict for LLM context (truncated abstract)."""
        return {
            "title": self.title,
            "authors": self.authors[:12],
            "year": self.year,
            "venue": self.venue,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "url": self.url,
            "abstract": (self.abstract or "")[:700],
            "source": self.source,
            "published_iso": self.published_iso,
        }


def _norm_title(title: str) -> str:
    return " ".join(title.lower().split())


def dedupe_candidates(candidates: list[CandidatePaper]) -> list[CandidatePaper]:
    """Drop duplicates by (normalized title, arxiv_id or doi)."""
    seen: set[tuple[str, str | None]] = set()
    out: list[CandidatePaper] = []
    for c in candidates:
        aid = (c.arxiv_id or "").strip() or None
        did = (c.doi or "").strip() or None
        key_id = aid or did
        key = (_norm_title(c.title), key_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _arxiv_id_from_entry_id(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"arxiv\.org/abs/([^?\s#]+)", text, re.I)
    return m.group(1).rstrip("/") if m else None


def search_arxiv(
    query: str,
    *,
    max_results: int,
    timeout: float,
) -> list[CandidatePaper]:
    """Query arXiv Atom API (``all:`` search)."""
    if max_results < 1:
        return []
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 100),
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    logger.info("arXiv search: %s", url[:120])
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = _request_with_retries(client, "GET", url)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            _mark_rate_limited("arxiv")
        logger.warning("arXiv search failed: %s", e)
        return []
    except httpx.HTTPError as e:
        logger.warning("arXiv search failed: %s", e)
        return []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        logger.warning("arXiv search invalid XML: %s", e)
        return []

    out: list[CandidatePaper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title_el = entry.find(f"{_ATOM}title")
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""
        summary_el = entry.find(f"{_ATOM}summary")
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""

        authors: list[str] = []
        for author in entry.findall(f"{_ATOM}author"):
            name_el = author.find(f"{_ATOM}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        published_el = entry.find(f"{_ATOM}published")
        published_iso: str | None = None
        year: int | None = None
        if published_el is not None and published_el.text:
            published_iso = published_el.text.strip()
            try:
                year = int(published_iso[:4])
            except ValueError:
                year = None

        id_el = entry.find(f"{_ATOM}id")
        abs_url = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = _arxiv_id_from_entry_id(abs_url)

        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("type") == "application/pdf" and link.get("href"):
                pdf_url = link.get("href", "").strip()
                break
        url = pdf_url or abs_url or None

        prim = entry.find(f"{_ARXIV_NS}primary_category")
        venue = prim.get("term") if prim is not None else "arXiv"

        if title:
            out.append(
                CandidatePaper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    url=url,
                    arxiv_id=arxiv_id,
                    doi=None,
                    venue=venue,
                    source="arxiv",
                    published_iso=published_iso,
                )
            )
    return out


def search_semantic_scholar(
    query: str,
    *,
    max_results: int,
    timeout: float,
    api_key: str | None,
) -> list[CandidatePaper]:
    """Semantic Scholar paper search (public tier; optional API key)."""
    if max_results < 1:
        return []
    fields = (
        "title,authors,year,abstract,url,externalIds,venue,journal,publicationVenue,publicationDate"
    )
    params: dict[str, str | int] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": fields,
    }
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key.strip()

    logger.info("Semantic Scholar search (limit=%s)", params["limit"])
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = _request_with_retries(
                client,
                "GET",
                SEMANTIC_SCHOLAR_SEARCH,
                params=params,
                headers=headers,
            )
            data = r.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            _mark_rate_limited("semantic_scholar")
        logger.warning("Semantic Scholar search failed: %s", e)
        return []
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Semantic Scholar search failed: %s", e)
        return []

    out: list[CandidatePaper] = []
    for item in data.get("data") or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        authors_raw = item.get("authors") or []
        authors = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]
        authors = [a for a in authors if a]
        year = item.get("year")
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None
        abstract = (item.get("abstract") or "").strip()
        url = (item.get("url") or "").strip() or None
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv") if isinstance(ext, dict) else None
        doi = ext.get("DOI") if isinstance(ext, dict) else None
        venue: str | None = None
        v = item.get("venue")
        if isinstance(v, dict):
            venue = (v.get("name") or "").strip() or None
        elif v:
            venue = str(v).strip() or None
        if not venue and item.get("journal"):
            j = item["journal"]
            jn = j.get("name", "") if isinstance(j, dict) else str(j)
            venue = jn.strip() or None

        pub_date = item.get("publicationDate")
        published_iso = str(pub_date).strip() if pub_date else None

        out.append(
            CandidatePaper(
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=url,
                arxiv_id=arxiv_id,
                doi=doi,
                venue=venue,
                source="semantic_scholar",
                published_iso=published_iso,
            )
        )
    return out


def search_literature(
    topic: str,
    *,
    max_results: int,
    settings: Settings | None = None,
) -> list[CandidatePaper]:
    """
    Search arXiv and Semantic Scholar, merge, dedupe, and cap to ``max_results``.

    Splits the budget roughly evenly between sources (when both succeed).

    Raises
    ------
    LiteratureSearchRateLimitError
        When both providers return no results and at least one appears rate-limited.
    """
    from core.config import load_settings

    settings = settings or load_settings()
    t = topic.strip()
    if not t:
        return []

    half = max(1, max_results // 2)
    n_arx = min(half, max_results)
    n_ss = max_results - n_arx
    timeout = settings.http_timeout
    _last_search_rate_limited_sources.clear()

    arx = search_arxiv(t, max_results=n_arx, timeout=timeout)
    ss = search_semantic_scholar(
        t,
        max_results=n_ss,
        timeout=timeout,
        api_key=settings.semantic_scholar_api_key,
    )

    merged = arx + ss
    merged = dedupe_candidates(merged)
    if merged:
        return merged[:max_results]

    if _last_search_rate_limited_sources:
        names = ", ".join(sorted(_last_search_rate_limited_sources))
        raise LiteratureSearchRateLimitError(
            "Literature providers are rate-limiting requests right now "
            f"({names}). Wait a bit and retry. If Semantic Scholar is involved, "
            "configure `SEMANTIC_SCHOLAR_API_KEY` to improve quota."
        )

    return merged[:max_results]
