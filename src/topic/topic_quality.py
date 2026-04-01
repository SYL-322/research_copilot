"""Heuristic quality signals for topic-scan routing and persistence (MVP, no extra LLM)."""

from __future__ import annotations

from typing import Any

from core.models import TopicReportLlmOutput
from topic.literature_search import CandidatePaper

# Length thresholds (characters, stripped)
RF_MIN_STRUCTURAL = 40
RF_MIN_STRONG = 80
TS_WEAK = 50


def prior_claims_from_previous_reports(
    previous_rows: list[dict[str, Any]],
    *,
    limit: int = 3,
    max_chars: int = 280,
) -> list[str]:
    """
    Extract up to ``limit`` short summary bullets from stored topic snapshots (no full JSON).
    """
    out: list[str] = []
    for row in previous_rows[:limit]:
        rep = row.get("report")
        if not isinstance(rep, dict):
            continue
        ts = (rep.get("topic_summary") or "").strip()
        if ts:
            s = ts[:max_chars]
            if len(ts) > max_chars:
                s = s.rstrip() + "..."
            out.append(s)
            continue
        rf = (rep.get("research_frontier") or "").strip()
        if rf:
            s = rf[:max_chars]
            if len(rf) > max_chars:
                s = s.rstrip() + "..."
            out.append(s)
        if len(out) >= limit:
            break
    return out[:limit]


def complexity_high(
    n_candidates: int,
    n_paper_memories: int,
    n_previous_topic_snapshots: int,
) -> bool:
    """Signals a heavier context; prefer a stronger model pass when combined with other triggers."""
    return (
        n_candidates > 20
        or n_paper_memories > 3
        or n_previous_topic_snapshots > 0
    )


def _count_grounded_mentions(llm: TopicReportLlmOutput) -> int:
    return (
        len(llm.foundational_papers)
        + len(llm.representative_papers)
        + len(llm.recent_valuable_papers)
        + len(llm.lower_priority_or_overhyped)
    )


def evaluate_topic_report(
    report: TopicReportLlmOutput,
    candidates: list[CandidatePaper],
) -> dict[str, Any]:
    """
    Return a lightweight quality assessment (no LLM).

    Keys: ``quality_score`` (0..1), ``is_low_quality`` (bool), ``issues`` (list[str]).
    """
    issues: list[str] = []
    score = 1.0

    md = (report.missing_directions or [])[:]
    rf = (report.research_frontier or "").strip()
    ts = (report.topic_summary or "").strip()
    n_cand = len(candidates)
    mentions = _count_grounded_mentions(report)

    if not md:
        issues.append("missing_directions_empty")
        score -= 0.35
    if len(rf) < RF_MIN_STRUCTURAL:
        issues.append("research_frontier_short")
        score -= 0.3
    elif len(rf) < RF_MIN_STRONG:
        issues.append("research_frontier_below_strong_threshold")
        score -= 0.12

    if mentions == 0 and n_cand > 0:
        issues.append("no_candidate_papers_grounded")
        score -= 0.35
    elif n_cand >= 10 and mentions < 3:
        issues.append("few_paper_mentions_vs_candidates")
        score -= 0.12

    if len(ts) < TS_WEAK and mentions >= 8:
        issues.append("summary_short_vs_many_mentions")
        score -= 0.1

    ev = (report.evidence_quality_note or "").lower()
    if "contradict" in ev or "inconsistent" in ev:
        issues.append("possible_contradiction_flagged")
        score -= 0.08

    score = max(0.0, min(1.0, score))

    is_low = (
        score < 0.6
        or not md
        or len(rf) < RF_MIN_STRUCTURAL
        or (mentions == 0 and n_cand > 0)
    )

    return {
        "quality_score": round(score, 3),
        "is_low_quality": is_low,
        "issues": issues,
    }


def report_meets_persistence_bar(
    report: TopicReportLlmOutput,
    *,
    quality_score: float,
    min_score: float = 0.5,
) -> tuple[bool, str]:
    if quality_score < min_score:
        return False, f"quality_score<{min_score}"
    if not (report.missing_directions or []):
        return False, "missing_directions_empty"
    rf = (report.research_frontier or "").strip()
    if len(rf) < RF_MIN_STRUCTURAL:
        return False, "research_frontier_trivial"
    return True, "ok"
