"""Find recent papers for digest topics (date filter + cross-topic deduplication)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from core.config import Settings
from topic.literature_search import (
    CandidatePaper,
    dedupe_candidates,
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


def find_recent_for_topic(
    topic: str,
    *,
    days_back: int,
    max_per_topic: int,
    settings: Settings,
    fetch_cap: int = 80,
) -> list[CandidatePaper]:
    """
    Search arXiv + Semantic Scholar for ``topic``, keep papers whose publication
    date is within the last ``days_back`` days (UTC calendar), newest first.

    Fetches up to ``fetch_cap`` per source before filtering (APIs do not always
    expose tight date filters).
    """
    t = topic.strip()
    if not t or max_per_topic < 1:
        return []

    cutoff = _utc_today() - timedelta(days=max(0, days_back))
    timeout = settings.http_timeout

    half = max(1, fetch_cap // 2)
    arx = search_arxiv(t, max_results=half, timeout=timeout)
    ss = search_semantic_scholar(
        t,
        max_results=fetch_cap - half,
        timeout=timeout,
        api_key=settings.semantic_scholar_api_key,
    )
    merged = dedupe_candidates(arx + ss)
    recent = [p for p in merged if _is_within_window(p, cutoff)]
    recent.sort(key=_sort_key_newest, reverse=True)
    out = recent[:max_per_topic]
    logger.info(
        "Topic %r: %d recent papers within %d days (from %d candidates)",
        t[:60],
        len(out),
        days_back,
        len(merged),
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
