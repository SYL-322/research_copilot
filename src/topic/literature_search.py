"""Search arXiv and Semantic Scholar for topic-relevant papers."""

from __future__ import annotations

import logging
import math
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
ARXIV_MIN_INTERVAL_SECONDS: Final = 3.1
ARXIV_RATE_LIMIT_COOLDOWN_SECONDS: Final = 300.0
SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS: Final = 1.05
_last_search_rate_limited_sources: set[str] = set()
_last_arxiv_request_monotonic: float | None = None
_arxiv_cooldown_until_monotonic: float | None = None
_last_semantic_scholar_request_monotonic: float | None = None


class LiteratureSearchRateLimitError(RuntimeError):
    """Raised when literature search providers are temporarily rate-limiting us."""


@dataclass
class ProviderSearchResponse:
    """Result of one provider request."""

    papers: list["CandidatePaper"] = field(default_factory=list)
    error: str | None = None
    rate_limited: bool = False


@dataclass
class ProviderQueryStat:
    """Per-provider retrieval accounting for transparency and debugging."""

    provider: str
    query_variants: list[str] = field(default_factory=list)
    raw_results: int = 0
    unique_results: int = 0
    errors: list[str] = field(default_factory=list)
    rate_limited: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "query_variants": list(self.query_variants),
            "raw_results": self.raw_results,
            "unique_results": self.unique_results,
            "errors": list(self.errors),
            "rate_limited": self.rate_limited,
        }


@dataclass
class LiteratureSearchResult:
    """Merged topic-search results with provider stats and candidate metadata."""

    topic: str
    normalized_topic: str
    query_variants: list[str]
    candidates: list["CandidatePaper"] = field(default_factory=list)
    provider_stats: list[ProviderQueryStat] = field(default_factory=list)
    raw_candidates: int = 0
    deduped_candidates: int = 0
    final_candidates: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "normalized_topic": self.normalized_topic,
            "query_variants": list(self.query_variants),
            "raw_candidates": self.raw_candidates,
            "deduped_candidates": self.deduped_candidates,
            "final_candidates": self.final_candidates,
            "provider_stats": [stat.as_dict() for stat in self.provider_stats],
        }


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
    if source == "arxiv":
        global _arxiv_cooldown_until_monotonic
        _arxiv_cooldown_until_monotonic = time.monotonic() + ARXIV_RATE_LIMIT_COOLDOWN_SECONDS


def _wait_for_semantic_scholar_slot() -> None:
    global _last_semantic_scholar_request_monotonic
    now = time.monotonic()
    if _last_semantic_scholar_request_monotonic is not None:
        elapsed = now - _last_semantic_scholar_request_monotonic
        remaining = SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            logger.info("Semantic Scholar throttle: sleeping %.2fs", remaining)
            time.sleep(remaining)
            now = time.monotonic()
    _last_semantic_scholar_request_monotonic = now


def _wait_for_arxiv_slot() -> None:
    global _last_arxiv_request_monotonic
    now = time.monotonic()
    if _last_arxiv_request_monotonic is not None:
        elapsed = now - _last_arxiv_request_monotonic
        remaining = ARXIV_MIN_INTERVAL_SECONDS - elapsed
        if remaining > 0:
            logger.info("arXiv throttle: sleeping %.2fs", remaining)
            time.sleep(remaining)
            now = time.monotonic()
    _last_arxiv_request_monotonic = now


def _should_retry_http_error(exc: httpx.HTTPStatusError) -> bool:
    code = exc.response.status_code
    return code == 429 or 500 <= code < 600


def _arxiv_in_cooldown() -> float | None:
    global _arxiv_cooldown_until_monotonic
    if _arxiv_cooldown_until_monotonic is None:
        return None
    remaining = _arxiv_cooldown_until_monotonic - time.monotonic()
    if remaining <= 0:
        _arxiv_cooldown_until_monotonic = None
        return None
    return remaining


def _request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, str | int] | None = None,
    headers: dict[str, str] | None = None,
    retry_on_429: bool = True,
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
            should_retry = _should_retry_http_error(exc)
            if exc.response.status_code == 429 and not retry_on_429:
                should_retry = False
            if attempt >= attempts - 1 or not should_retry:
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
    source_signals: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)
    retrieval_rank: int | None = None
    topic_relevance_score: float = 0.0
    topic_relevance_reasons: list[str] = field(default_factory=list)

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
            "source_signals": self.source_signals,
            "matched_queries": self.matched_queries,
            "retrieval_rank": self.retrieval_rank,
            "published_iso": self.published_iso,
        }


def _norm_title(title: str) -> str:
    return " ".join(title.lower().split())


_GENERIC_TOPIC_TERMS: Final[frozenset[str]] = frozenset(
    {
        "analysis",
        "approach",
        "approaches",
        "benchmark",
        "benchmarks",
        "data",
        "dataset",
        "datasets",
        "framework",
        "frameworks",
        "method",
        "methods",
        "model",
        "models",
        "study",
        "studies",
        "survey",
        "surveys",
        "system",
        "systems",
    }
)


def _tokenize_topic_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def _normalize_match_text(text: str) -> str:
    tokens = _tokenize_topic_text(text)
    return " ".join(tokens)


def _topic_terms(topic: str) -> list[str]:
    return _tokenize_topic_text(_normalize_topic_query(topic))


def _topic_informative_terms(topic: str) -> list[str]:
    terms = _topic_terms(topic)
    informative = [term for term in terms if term not in _GENERIC_TOPIC_TERMS]
    return informative or terms


def _normalize_topic_query(topic: str) -> str:
    """Normalize topic strings before query expansion."""
    text = (topic or "").strip()
    if not text:
        return ""
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("•", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:")


def _strip_parenthetical(text: str) -> str:
    stripped = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", stripped).strip(" ,;:-")


def _looks_atomic(segment: str) -> bool:
    words = [w for w in re.split(r"\s+", segment) if w]
    return 0 < len(words) <= 5


def _combine_shared_prefix(longer: str, shorter: str) -> str | None:
    longer_tokens = [tok for tok in longer.split() if tok]
    shorter_tokens = [tok for tok in shorter.split() if tok]
    if len(longer_tokens) < 2 or not shorter_tokens or len(shorter_tokens) > 3:
        return None
    prefix = longer_tokens[:-1]
    if not prefix:
        return None
    candidate = " ".join(prefix + shorter_tokens)
    if candidate.lower() == longer.lower():
        return None
    return candidate


def expand_topic_queries(topic: str, *, max_queries: int = 6) -> list[str]:
    """Generate multiple recall-oriented query variants for a topic string."""
    base = _normalize_topic_query(topic)
    if not base:
        return []

    variants: list[str] = []

    def add(text: str) -> None:
        q = _normalize_topic_query(text)
        if not q:
            return
        key = q.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(q)

    seen: set[str] = set()
    add(base)

    no_paren = _strip_parenthetical(base)
    if no_paren and no_paren.casefold() != base.casefold():
        add(no_paren)

    splitter = re.compile(r"\s*(?:/|\band\b|\bor\b|\s-\s)\s*", re.IGNORECASE)
    parts = [p.strip(" ,;:") for p in splitter.split(no_paren or base) if p.strip(" ,;:")]
    if 1 < len(parts) <= 4:
        combined = " ".join(parts)
        add(combined)
        for part in parts:
            if _looks_atomic(part):
                add(part)
        if len(parts) == 2:
            left, right = parts
            add(_combine_shared_prefix(left, right) or "")
            add(_combine_shared_prefix(right, left) or "")

    replacement_patterns = (
        re.compile(r"/"),
        re.compile(r"\sand\s", re.IGNORECASE),
        re.compile(r"\s-\s"),
    )
    for pattern in replacement_patterns:
        if pattern.search(base):
            add(pattern.sub(" ", base))

    return variants[:max_queries]


def _build_arxiv_query(variant: str) -> str:
    """Build a stricter arXiv expression for short phrase-like queries.

    For multi-word topics we avoid relying on a raw ``all:<topic>`` search because
    high-frequency terms such as ``dataset`` create many recent but irrelevant hits.
    The query prefers title/abstract phrase matches, then falls back to a
    conjunctive title/abstract term match so every topic term must appear.
    """
    normalized = _normalize_topic_query(variant)
    terms = _topic_terms(normalized)
    if len(terms) < 2:
        return f"all:{normalized}"

    phrase = _normalize_match_text(normalized)
    exact = f'ti:"{phrase}" OR abs:"{phrase}"'
    conjunctive = " AND ".join(f"(ti:{term} OR abs:{term})" for term in terms)
    return f"({exact}) OR ({conjunctive})"


def _candidate_identity_keys(candidate: CandidatePaper) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if candidate.arxiv_id:
        keys.append(("arxiv", candidate.arxiv_id.strip().lower()))
    if candidate.doi:
        keys.append(("doi", candidate.doi.strip().lower()))
    title = _norm_title(candidate.title)
    if title:
        keys.append(("title", title))
    return keys


def _prefer_candidate_metadata(existing: CandidatePaper, incoming: CandidatePaper) -> CandidatePaper:
    """Choose the richer metadata row while preserving merged provenance."""
    existing_score = (
        len(existing.abstract or ""),
        len(existing.authors),
        1 if existing.url else 0,
        1 if existing.year else 0,
    )
    incoming_score = (
        len(incoming.abstract or ""),
        len(incoming.authors),
        1 if incoming.url else 0,
        1 if incoming.year else 0,
    )
    return incoming if incoming_score > existing_score else existing


def dedupe_candidates(candidates: list[CandidatePaper]) -> list[CandidatePaper]:
    """Merge duplicates by arXiv id, DOI, or normalized title."""
    merged: list[CandidatePaper] = []
    key_to_index: dict[tuple[str, str], int] = {}

    for candidate in candidates:
        match_index: int | None = None
        for key in _candidate_identity_keys(candidate):
            if key in key_to_index:
                match_index = key_to_index[key]
                break

        if match_index is None:
            clone = CandidatePaper(**candidate.__dict__)
            clone.source_signals = list(dict.fromkeys(candidate.source_signals or [candidate.source]))
            clone.matched_queries = list(dict.fromkeys(candidate.matched_queries))
            merged.append(clone)
            idx = len(merged) - 1
            for key in _candidate_identity_keys(clone):
                key_to_index[key] = idx
            continue

        existing = merged[match_index]
        preferred = _prefer_candidate_metadata(existing, candidate)
        if preferred is candidate:
            replacement = CandidatePaper(**candidate.__dict__)
            replacement.source_signals = list(
                dict.fromkeys((existing.source_signals or [existing.source]) + (candidate.source_signals or [candidate.source]))
            )
            replacement.matched_queries = list(
                dict.fromkeys(existing.matched_queries + candidate.matched_queries)
            )
            merged[match_index] = replacement
            existing = replacement
        else:
            existing.source_signals = list(
                dict.fromkeys((existing.source_signals or [existing.source]) + (candidate.source_signals or [candidate.source]))
            )
            existing.matched_queries = list(
                dict.fromkeys(existing.matched_queries + candidate.matched_queries)
            )
            if not existing.venue and candidate.venue:
                existing.venue = candidate.venue
            if not existing.published_iso and candidate.published_iso:
                existing.published_iso = candidate.published_iso
            if not existing.url and candidate.url:
                existing.url = candidate.url
            if not existing.year and candidate.year:
                existing.year = candidate.year

        for key in _candidate_identity_keys(existing):
            key_to_index[key] = match_index

    return merged


def _candidate_sort_key(candidate: CandidatePaper) -> tuple[float, ...]:
    year = float(candidate.year or 0)
    has_abstract = 1.0 if (candidate.abstract or "").strip() else 0.0
    provider_count = float(len(candidate.source_signals or [candidate.source]))
    query_hits = float(len(candidate.matched_queries))
    rank_bonus = (
        0.0 if candidate.retrieval_rank is None else max(0.0, 50.0 - candidate.retrieval_rank)
    )
    return (
        candidate.topic_relevance_score,
        provider_count,
        query_hits,
        year,
        has_abstract,
        rank_bonus,
    )


def _score_topic_relevance(topic: str, candidate: CandidatePaper) -> tuple[float, list[str]]:
    """Compute a lightweight lexical relevance score for merged topic candidates.

    The score is intentionally transparent and field-aware:
    exact phrase matches and title coverage dominate, abstract coverage helps,
    and generic-only matches receive a penalty. Provider count and recency are
    still used later, but only as tie-breakers after relevance.
    """
    normalized_topic = _normalize_topic_query(topic)
    terms = _topic_terms(normalized_topic)
    if not terms:
        return 0.0, []

    informative_terms = _topic_informative_terms(normalized_topic)
    phrase = _normalize_match_text(normalized_topic)
    title_text = _normalize_match_text(candidate.title)
    abstract_text = _normalize_match_text(candidate.abstract)
    title_terms = set(_tokenize_topic_text(candidate.title))
    abstract_terms = set(_tokenize_topic_text(candidate.abstract))
    combined_terms = title_terms | abstract_terms

    score = 0.0
    reasons: list[str] = []

    if len(terms) >= 2 and phrase and phrase in title_text:
        score += 8.0
        reasons.append("exact_phrase:title")
    elif len(terms) >= 2 and phrase and phrase in abstract_text:
        score += 4.0
        reasons.append("exact_phrase:abstract")

    title_matches = [term for term in terms if term in title_terms]
    abstract_only_matches = [
        term for term in terms if term not in title_terms and term in abstract_terms
    ]
    informative_matches = [term for term in informative_terms if term in combined_terms]

    if title_matches:
        coverage = len(title_matches) / len(terms)
        score += 5.0 * coverage
        reasons.append(f"title_terms:{len(title_matches)}/{len(terms)}")
    if abstract_only_matches:
        coverage = len(abstract_only_matches) / len(terms)
        score += 2.5 * coverage
        reasons.append(f"abstract_terms:{len(abstract_only_matches)}/{len(terms)}")
    if len(title_matches) == len(terms):
        score += 4.0
        reasons.append("all_terms:title")
    elif len(combined_terms.intersection(terms)) == len(terms):
        score += 2.0
        reasons.append("all_terms:combined")
    if informative_matches:
        score += 3.0 * (len(informative_matches) / len(informative_terms))
        reasons.append(f"informative_terms:{len(informative_matches)}/{len(informative_terms)}")

    generic_only_matches = [term for term in terms if term in combined_terms and term not in informative_terms]
    if generic_only_matches and not informative_matches:
        score -= 3.0
        reasons.append("generic_only_penalty")
    if len(terms) >= 2 and len(combined_terms.intersection(terms)) <= 1 and phrase not in title_text:
        score -= 1.5
        reasons.append("low_coverage_penalty")

    return score, reasons


def _passes_topic_relevance_gate(topic: str, candidate: CandidatePaper) -> bool:
    """Drop obviously off-topic multi-word hits while keeping single strong matches.

    This is conservative on purpose: a candidate stays if it hits the full phrase,
    covers all topic terms somewhere, or has an informative title hit. The goal is
    to remove papers that only match a generic term like ``dataset``.
    """
    terms = _topic_terms(topic)
    if len(terms) < 2:
        return True

    phrase = _normalize_match_text(topic)
    title_text = _normalize_match_text(candidate.title)
    abstract_text = _normalize_match_text(candidate.abstract)
    title_terms = set(_tokenize_topic_text(candidate.title))
    abstract_terms = set(_tokenize_topic_text(candidate.abstract))
    combined_terms = title_terms | abstract_terms
    informative_terms = set(_topic_informative_terms(topic))
    informative_title_hits = informative_terms.intersection(title_terms)

    if phrase and (phrase in title_text or phrase in abstract_text):
        return True
    if len(combined_terms.intersection(terms)) == len(terms):
        return True
    if informative_title_hits and len(combined_terms.intersection(terms)) >= 1:
        return True
    return False


def _rank_and_filter_candidates(topic: str, candidates: list[CandidatePaper]) -> list[CandidatePaper]:
    """Apply transparent topic relevance scoring before the existing tie-breakers."""
    scored: list[CandidatePaper] = []
    filtered_out: list[CandidatePaper] = []

    for candidate in candidates:
        score, reasons = _score_topic_relevance(topic, candidate)
        candidate.topic_relevance_score = score
        candidate.topic_relevance_reasons = reasons
        if _passes_topic_relevance_gate(topic, candidate):
            scored.append(candidate)
        else:
            filtered_out.append(candidate)

    ranked = sorted(scored, key=_candidate_sort_key, reverse=True)
    if filtered_out:
        logger.info(
            "Filtered %d low-relevance candidate(s) for topic %r: %s",
            len(filtered_out),
            topic,
            "; ".join(
                f"{paper.title[:80]} (score={paper.topic_relevance_score:.2f}, reasons={paper.topic_relevance_reasons})"
                for paper in filtered_out[:5]
            ),
        )
    if ranked:
        logger.info(
            "Top topic candidates for %r: %s",
            topic,
            "; ".join(
                f"{paper.title[:80]} (score={paper.topic_relevance_score:.2f}, reasons={paper.topic_relevance_reasons})"
                for paper in ranked[:5]
            ),
        )
        return ranked
    return sorted(candidates, key=_candidate_sort_key, reverse=True)


def _arxiv_id_from_entry_id(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"arxiv\.org/abs/([^?\s#]+)", text, re.I)
    return m.group(1).rstrip("/") if m else None


def _search_arxiv_response(
    query: str,
    *,
    max_results: int,
    timeout: float,
) -> ProviderSearchResponse:
    """Query arXiv Atom API using a field-aware expression."""
    if max_results < 1:
        return ProviderSearchResponse()
    if (cooldown_remaining := _arxiv_in_cooldown()) is not None:
        logger.warning(
            "arXiv search skipped for query %r due to active cooldown (%.1fs remaining).",
            query,
            cooldown_remaining,
        )
        _mark_rate_limited("arxiv")
        return ProviderSearchResponse(error="rate_limited_cooldown", rate_limited=True)
    arxiv_query = _build_arxiv_query(query)
    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": min(max_results, 100),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    logger.info("arXiv search for %r using %s", query, arxiv_query)
    try:
        _wait_for_arxiv_slot()
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = _request_with_retries(client, "GET", url, retry_on_429=False)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            _mark_rate_limited("arxiv")
            return ProviderSearchResponse(
                error="rate_limited",
                rate_limited=True,
            )
        logger.warning("arXiv search failed: %s", e)
        return ProviderSearchResponse(error=f"http_{e.response.status_code}")
    except httpx.HTTPError as e:
        logger.warning("arXiv search failed: %s", e)
        return ProviderSearchResponse(error=e.__class__.__name__)

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        logger.warning("arXiv search invalid XML: %s", e)
        return ProviderSearchResponse(error="invalid_xml")

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
                    source_signals=["arxiv"],
                    matched_queries=[query],
                )
            )
    return ProviderSearchResponse(papers=out)


def search_arxiv(
    query: str,
    *,
    max_results: int,
    timeout: float,
) -> list[CandidatePaper]:
    return _search_arxiv_response(query, max_results=max_results, timeout=timeout).papers


def _search_semantic_scholar_response(
    query: str,
    *,
    max_results: int,
    timeout: float,
    api_key: str | None,
) -> ProviderSearchResponse:
    """Semantic Scholar paper search (public tier; optional API key)."""
    if max_results < 1:
        return ProviderSearchResponse()
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
        _wait_for_semantic_scholar_slot()
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
            return ProviderSearchResponse(
                error="rate_limited",
                rate_limited=True,
            )
        logger.warning("Semantic Scholar search failed: %s", e)
        return ProviderSearchResponse(error=f"http_{e.response.status_code}")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Semantic Scholar search failed: %s", e)
        return ProviderSearchResponse(error=e.__class__.__name__)

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
                source_signals=["semantic_scholar"],
                matched_queries=[query],
            )
        )
    return ProviderSearchResponse(papers=out)


def search_semantic_scholar(
    query: str,
    *,
    max_results: int,
    timeout: float,
    api_key: str | None,
) -> list[CandidatePaper]:
    return _search_semantic_scholar_response(
        query,
        max_results=max_results,
        timeout=timeout,
        api_key=api_key,
    ).papers


def _search_provider_variants(
    provider: str,
    query_variants: list[str],
    *,
    max_results: int,
    timeout: float,
    api_key: str | None = None,
) -> tuple[list[CandidatePaper], ProviderQueryStat]:
    stat = ProviderQueryStat(provider=provider, query_variants=list(query_variants))
    out: list[CandidatePaper] = []
    per_query_limit = min(25, max(8, math.ceil(max_results * 0.75)))
    recovered_after_rate_limit = False

    for query in query_variants:
        if provider == "arxiv":
            response = _search_arxiv_response(query, max_results=per_query_limit, timeout=timeout)
        else:
            response = _search_semantic_scholar_response(
                query,
                max_results=per_query_limit,
                timeout=timeout,
                api_key=api_key,
            )

        stat.raw_results += len(response.papers)
        if response.error:
            if response.error not in stat.errors:
                stat.errors.append(response.error)
        stat.rate_limited = stat.rate_limited or response.rate_limited
        if response.rate_limited:
            if recovered_after_rate_limit:
                logger.warning(
                    "%s rate-limited again on query variant %r after cooldown; skipping remaining variants for this provider.",
                    provider,
                    query,
                )
                break
            cooldown = 6.0 if provider == "arxiv" else 3.0
            logger.warning(
                "%s rate-limited on query variant %r; cooling down %.1fs before continuing remaining variants.",
                provider,
                query,
                cooldown,
            )
            time.sleep(cooldown)
            recovered_after_rate_limit = True
            continue

        for idx, paper in enumerate(response.papers):
            paper.source_signals = list(dict.fromkeys((paper.source_signals or [provider]) + [provider]))
            if query not in paper.matched_queries:
                paper.matched_queries.append(query)
            paper.retrieval_rank = idx
            out.append(paper)

    unique = dedupe_candidates(out)
    stat.unique_results = len(unique)
    return unique, stat


def search_literature_detailed(
    topic: str,
    *,
    max_results: int,
    settings: Settings | None = None,
) -> LiteratureSearchResult:
    """Search multiple query variants and return candidates plus retrieval accounting."""
    from core.config import load_settings

    settings = settings or load_settings()
    normalized = _normalize_topic_query(topic)
    query_variants = expand_topic_queries(normalized)
    if not normalized or not query_variants:
        return LiteratureSearchResult(
            topic=topic,
            normalized_topic=normalized,
            query_variants=[],
        )

    timeout = settings.http_timeout
    _last_search_rate_limited_sources.clear()

    arxiv_candidates, arxiv_stat = _search_provider_variants(
        "arxiv",
        query_variants,
        max_results=max_results,
        timeout=timeout,
    )
    ss_candidates, ss_stat = _search_provider_variants(
        "semantic_scholar",
        query_variants,
        max_results=max_results,
        timeout=timeout,
        api_key=settings.semantic_scholar_api_key,
    )

    merged = dedupe_candidates(arxiv_candidates + ss_candidates)
    ranked = _rank_and_filter_candidates(normalized, merged)
    final = ranked[:max_results]

    rate_limited = [stat.provider for stat in (arxiv_stat, ss_stat) if stat.rate_limited]
    if not final and rate_limited:
        names = ", ".join(sorted(set(rate_limited) | _last_search_rate_limited_sources))
        raise LiteratureSearchRateLimitError(
            "Literature providers are rate-limiting requests right now "
            f"({names}). Wait a bit and retry. If Semantic Scholar is involved, "
            "configure `SEMANTIC_SCHOLAR_API_KEY` to improve quota."
        )

    return LiteratureSearchResult(
        topic=topic.strip(),
        normalized_topic=normalized,
        query_variants=query_variants,
        candidates=final,
        provider_stats=[arxiv_stat, ss_stat],
        raw_candidates=arxiv_stat.raw_results + ss_stat.raw_results,
        deduped_candidates=len(merged),
        final_candidates=len(final),
    )


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
    result = search_literature_detailed(topic, max_results=max_results, settings=settings)
    if result.candidates:
        return result.candidates
    return result.candidates
