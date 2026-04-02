"""Find recent papers for digest topics (relevance gate + date filter + deduplication)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from core.config import Settings
from topic.literature_search import (
    CandidatePaper,
    dedupe_candidates,
    expand_topic_queries,
    rank_and_filter_topic_candidates,
    search_arxiv,
    search_semantic_scholar,
)

logger = logging.getLogger(__name__)


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def parse_publication_date(p: CandidatePaper) -> date | None:
    """Best-effort publication date from metadata."""
    if p.published_iso:
        raw = p.published_iso.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            pass
        try:
            return date.fromisoformat(p.published_iso.strip()[:10])
        except ValueError:
            pass
    if p.year is not None:
        try:
            return date(int(p.year), 1, 1)
        except (TypeError, ValueError):
            pass
    return None


def _is_within_window(p: CandidatePaper, cutoff: date) -> bool:
    d = parse_publication_date(p)
    if d is None:
        return False
    return d >= cutoff


def _sort_key_newest(p: CandidatePaper) -> tuple[date, str]:
    d = parse_publication_date(p) or date.min
    return (d, p.title)


def _sort_key_recent_relevant(p: CandidatePaper) -> tuple[date, float, str]:
    """Digest ordering: recency first, then topic relevance among recent survivors."""
    d = parse_publication_date(p) or date.min
    return (d, p.topic_relevance_score, p.title)


def _query_budget_for_digest_variant(topic: str, query: str) -> int:
    """Assign a small per-variant fetch budget based on query specificity.

    Digest query expansion variants are not equally valuable. The original topic
    string and closely related multi-word combined forms are strongest, plain
    atomic terms are weaker recall helpers, and morphology-derived single-word
    expansions are the weakest. A fixed layered budget works better here than
    dividing one budget equally across all variants, which penalizes more
    expressive topics.
    """
    topic_norm = " ".join(topic.strip().lower().split())
    query_norm = " ".join(query.strip().lower().split())
    if query_norm == topic_norm:
        return 25
    if " " in query_norm:
        return 20
    if query_norm.endswith("ed"):
        return 8
    return 12


def _search_digest_topic_variants(
    topic: str,
    *,
    fetch_cap: int,
    settings: Settings,
) -> list[CandidatePaper]:
    """Search digest topics across lightweight query variants, then dedupe.

    This intentionally reuses the same topic-query expansion as topic scan, but
    keeps the digest flow local: provider fetches still happen here, and the
    downstream digest pipeline remains unchanged.
    """
    query_variants = expand_topic_queries(topic) or [topic]
    timeout = settings.http_timeout

    out: list[CandidatePaper] = []
    for query in query_variants:
        per_query_limit = min(fetch_cap, _query_budget_for_digest_variant(topic, query))
        out.extend(search_arxiv(query, max_results=per_query_limit, timeout=timeout))
        out.extend(
            search_semantic_scholar(
                query,
                max_results=per_query_limit,
                timeout=timeout,
                api_key=settings.semantic_scholar_api_key,
            )
        )
    return dedupe_candidates(out)


def find_recent_for_topic(
    topic: str,
    *,
    days_back: int,
    max_per_topic: int,
    settings: Settings,
    fetch_cap: int = 80,
) -> list[CandidatePaper]:
    """
    Search arXiv + Semantic Scholar for ``topic``, apply the lightweight topic
    relevance gate used by topic scan, then keep papers whose publication date
    is within the last ``days_back`` days (UTC calendar).

    Fetches up to ``fetch_cap`` per source before filtering (APIs do not always
    expose tight date filters).
    """
    t = topic.strip()
    if not t or max_per_topic < 1:
        return []

    cutoff = _utc_today() - timedelta(days=max(0, days_back))
    merged = _search_digest_topic_variants(
        t,
        fetch_cap=fetch_cap,
        settings=settings,
    )
    ranked = rank_and_filter_topic_candidates(
        t,
        merged,
        fallback_to_unfiltered=False,
    )
    recent = [p for p in ranked if _is_within_window(p, cutoff)]
    recent.sort(key=_sort_key_recent_relevant, reverse=True)
    out = recent[:max_per_topic]
    logger.info(
        "Topic %r: %d recent relevant papers within %d days (from %d merged candidates, %d after relevance gate)",
        t[:60],
        len(out),
        days_back,
        len(merged),
        len(ranked),
    )
    return out


def _paper_dedupe_key(p: CandidatePaper) -> tuple[str, str | None]:
    aid = (p.arxiv_id or "").strip() or None
    did = (p.doi or "").strip() or None
    key_id = aid or did
    title = " ".join(p.title.lower().split())
    return (title, key_id)


@dataclass
class TopicPaperBatch:
    """One paper with the subscription topics that surfaced it."""

    paper: CandidatePaper
    matched_topics: list[str] = field(default_factory=list)


@dataclass
class TopicRecentDebug:
    """Debug snapshot of one digest topic before and after candidate filtering."""

    topic: str
    query_variants: list[str] = field(default_factory=list)
    merged_candidates: list[CandidatePaper] = field(default_factory=list)
    relevance_candidates: list[CandidatePaper] = field(default_factory=list)
    recent_candidates: list[CandidatePaper] = field(default_factory=list)
    final_candidates: list[CandidatePaper] = field(default_factory=list)


def collect_recent_across_topics(
    topics: Iterable[str],
    *,
    days_back: int,
    max_per_topic: int,
    settings: Settings,
) -> list[TopicPaperBatch]:
    """
    For each topic, find recent papers, then merge duplicates across topics
    (same title + arXiv id / DOI).
    """
    buckets: dict[tuple[str, str | None], TopicPaperBatch] = {}
    for topic in topics:
        t = topic.strip()
        if not t:
            continue
        papers = find_recent_for_topic(
            t,
            days_back=days_back,
            max_per_topic=max_per_topic,
            settings=settings,
        )
        for p in papers:
            k = _paper_dedupe_key(p)
            if k not in buckets:
                buckets[k] = TopicPaperBatch(paper=p, matched_topics=[t])
            else:
                if t not in buckets[k].matched_topics:
                    buckets[k].matched_topics.append(t)

    out = list(buckets.values())
    out.sort(key=lambda b: _sort_key_newest(b.paper), reverse=True)
    logger.info("Cross-topic merge: %d unique papers", len(out))
    return out


def collect_recent_across_topics_with_debug(
    topics: Iterable[str],
    *,
    days_back: int,
    max_per_topic: int,
    settings: Settings,
    fetch_cap: int = 80,
) -> tuple[list[TopicPaperBatch], list[TopicRecentDebug]]:
    """Return digest batches plus per-topic candidate snapshots for debugging."""
    buckets: dict[tuple[str, str | None], TopicPaperBatch] = {}
    debug_rows: list[TopicRecentDebug] = []

    for topic in topics:
        t = topic.strip()
        if not t:
            continue
        cutoff = _utc_today() - timedelta(days=max(0, days_back))
        merged = _search_digest_topic_variants(
            t,
            fetch_cap=fetch_cap,
            settings=settings,
        )
        ranked = rank_and_filter_topic_candidates(
            t,
            merged,
            fallback_to_unfiltered=False,
        )
        recent = [p for p in ranked if _is_within_window(p, cutoff)]
        recent.sort(key=_sort_key_recent_relevant, reverse=True)
        final = recent[:max_per_topic]
        debug_rows.append(
            TopicRecentDebug(
                topic=t,
                query_variants=expand_topic_queries(t) or [t],
                merged_candidates=list(merged),
                relevance_candidates=list(ranked),
                recent_candidates=list(recent),
                final_candidates=list(final),
            )
        )
        for p in final:
            k = _paper_dedupe_key(p)
            if k not in buckets:
                buckets[k] = TopicPaperBatch(paper=p, matched_topics=[t])
            elif t not in buckets[k].matched_topics:
                buckets[k].matched_topics.append(t)

    out = list(buckets.values())
    out.sort(key=lambda b: _sort_key_newest(b.paper), reverse=True)
    logger.info("Cross-topic merge: %d unique papers", len(out))
    return out, debug_rows
