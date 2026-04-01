"""LLM synthesis of a structured topic report from search candidates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.config import Settings
from core.models import TopicPaperMention, TopicReport, TopicReportLlmOutput
from llm.openai_client import OpenAIClient, strip_json_fences
from topic.literature_search import CandidatePaper
from utils.text_normalize import normalize_title

logger = logging.getLogger(__name__)

PROMPT_FILE = "topic_scan_prompt.md"


class TopicAnalysisError(Exception):
    """Raised when the topic report cannot be synthesized."""


def load_topic_prompt(project_root: Path) -> str:
    path = project_root / "prompts" / PROMPT_FILE
    if not path.is_file():
        raise TopicAnalysisError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def synthesize_topic_report(
    topic: str,
    candidates: list[CandidatePaper],
    *,
    settings: Settings,
    project_root: Path,
    related_memories: list[dict[str, Any]] | None = None,
    prior_claims_summary: list[str] | None = None,
    initial_topic_report_json: str | None = None,
    model: str | None = None,
    temperature: float = 0.25,
) -> TopicReportLlmOutput:
    """
    Call the LLM with candidate metadata and optional local paper memories, then return a
    validated structured report.

    Raises
    ------
    TopicAnalysisError
        Empty topic, no candidates, or unrecoverable parse errors.
    """
    t = topic.strip()
    if not t:
        raise TopicAnalysisError("Topic is empty.")
    if not candidates:
        raise TopicAnalysisError("No candidate papers from literature search.")

    payload = [c.as_prompt_dict() for c in candidates]
    papers_json = json.dumps(payload, ensure_ascii=False, indent=2)
    memories = related_memories or []
    memories_json = json.dumps(memories, ensure_ascii=False, indent=2)
    prior_claims = prior_claims_summary or []
    prior_claims_json = json.dumps(prior_claims, ensure_ascii=False, indent=2)
    initial_slot = (
        initial_topic_report_json.strip()
        if initial_topic_report_json and initial_topic_report_json.strip()
        else "null"
    )
    refining = initial_slot not in ("null", "{}", "")

    template = load_topic_prompt(project_root)
    required = (
        "{{TOPIC}}",
        "{{CANDIDATE_PAPERS_JSON}}",
        "{{RELATED_PAPER_MEMORIES_JSON}}",
        "{{PRIOR_CLAIMS_SUMMARY_JSON}}",
        "{{INITIAL_TOPIC_REPORT_JSON}}",
    )
    for ph in required:
        if ph not in template:
            raise TopicAnalysisError(f"Prompt template must contain {ph}.")

    user_content = (
        template.replace("{{TOPIC}}", t)
        .replace("{{CANDIDATE_PAPERS_JSON}}", papers_json)
        .replace("{{RELATED_PAPER_MEMORIES_JSON}}", memories_json)
        .replace("{{PRIOR_CLAIMS_SUMMARY_JSON}}", prior_claims_json)
        .replace("{{INITIAL_TOPIC_REPORT_JSON}}", initial_slot)
    )

    client = OpenAIClient(settings)
    use_model = model or settings.resolve_openai_model_main()
    sys_parts = [
        "You output only valid JSON matching the user's schema. "
        "Do not include markdown fences. "
        "Every paper listed in foundational/representative/recent/lower_priority arrays "
        "must correspond to an entry in the candidate JSON. "
        "When paper memories are provided, prefer them over metadata for "
        "method detail, assumptions, design rationale, and limitations.",
    ]
    if prior_claims:
        sys_parts.append(
            " PRIOR_CLAIMS_SUMMARY_JSON lists short hypotheses from earlier topic snapshots—"
            "treat them as hypotheses, not facts; keep or reuse only what current candidates "
            "and paper memories support."
        )
    if refining:
        sys_parts.append(
            " Refinement pass: INITIAL_TOPIC_REPORT_JSON is the first-pass draft (same schema). "
            "Do NOT repeat that report. Focus on deeper comparison, better insights, and clearer gaps; "
            "especially improve research_frontier and missing_directions. Output one full replacement JSON."
        )
    messages = [
        {"role": "system", "content": "".join(sys_parts)},
        {"role": "user", "content": user_content},
    ]

    try:
        return client.chat_parse(
            messages,
            TopicReportLlmOutput,
            model=use_model,
            temperature=temperature,
        )
    except Exception as e:
        logger.warning("Structured parse failed (%s); trying json_object + repair", e)
        raw = client.chat_json_object(messages, model=use_model, temperature=temperature)
        try:
            return TopicReportLlmOutput.model_validate_json(strip_json_fences(raw))
        except ValidationError as e2:
            raise TopicAnalysisError(f"Invalid topic report JSON from model: {e2}") from e2


def _arxiv_key(aid: str | None) -> str | None:
    if not aid:
        return None
    return aid.strip().lower()


def _candidate_title_arxiv_sets(
    candidates: list[CandidatePaper],
) -> tuple[set[str], set[str]]:
    titles = {normalize_title(c.title) for c in candidates}
    arxiv = {a for c in candidates if c.arxiv_id and (a := _arxiv_key(c.arxiv_id))}
    return titles, arxiv


def _mention_matches_candidates(
    m: TopicPaperMention,
    valid_titles: set[str],
    valid_arxiv: set[str],
) -> bool:
    if m.arxiv_id and (a := _arxiv_key(m.arxiv_id)) and a in valid_arxiv:
        return True
    if normalize_title(m.title) in valid_titles:
        return True
    return False


def _filter_mentions(
    mentions: list[TopicPaperMention],
    candidates: list[CandidatePaper],
) -> list[TopicPaperMention]:
    vt, va = _candidate_title_arxiv_sets(candidates)
    out: list[TopicPaperMention] = []
    for m in mentions:
        if _mention_matches_candidates(m, vt, va):
            out.append(m)
        else:
            logger.warning(
                "Removed topic mention not in candidate set: %r",
                (m.title or "")[:120],
            )
    return out


def llm_output_to_topic_report(
    topic: str,
    llm: TopicReportLlmOutput,
    candidates: list[CandidatePaper],
) -> TopicReport:
    """Build a :class:`TopicReport` domain object from LLM output."""
    summary = (llm.topic_summary or "").strip()
    if len(summary) > 400:
        summary = summary[:397] + "..."
    return TopicReport(
        topic=topic.strip(),
        topic_summary=llm.topic_summary,
        branches_subthemes=list(llm.branches_subthemes),
        foundational_papers=_filter_mentions(list(llm.foundational_papers), candidates),
        representative_papers=_filter_mentions(list(llm.representative_papers), candidates),
        recent_valuable_papers=_filter_mentions(list(llm.recent_valuable_papers), candidates),
        lower_priority_or_overhyped=_filter_mentions(list(llm.lower_priority_or_overhyped), candidates),
        recent_trends=list(llm.recent_trends),
        open_questions=list(llm.open_questions),
        evidence_quality_note=llm.evidence_quality_note,
        missing_directions=list(llm.missing_directions),
        research_frontier=llm.research_frontier,
        cross_paper_insights=list(llm.cross_paper_insights),
        method_comparison_summary=llm.method_comparison_summary,
        evolution_notes=llm.evolution_notes,
        summary=summary or None,
    )


def render_topic_report_markdown(report: TopicReport) -> str:
    """Render a readable Markdown document from a structured topic report."""
    lines: list[str] = []
    lines.append(f"# Topic: {report.topic}\n")
    lines.append("## Summary\n")
    lines.append((report.topic_summary or "").strip() or "_No summary._")
    lines.append("")

    if (report.evolution_notes or "").strip():
        lines.append("## Evolution notes\n")
        lines.append(report.evolution_notes.strip())
        lines.append("")

    if report.evidence_quality_note:
        lines.append("## Evidence quality\n")
        lines.append(report.evidence_quality_note.strip())
        lines.append("")

    if (report.research_frontier or "").strip():
        lines.append("## Research frontier\n")
        lines.append(report.research_frontier.strip())
        lines.append("")

    if report.missing_directions:
        lines.append("## Missing directions (likely absent from candidate list)\n")
        for x in report.missing_directions:
            lines.append(f"- {x}")
        lines.append("")

    if (report.method_comparison_summary or "").strip():
        lines.append("## Method comparison\n")
        lines.append(report.method_comparison_summary.strip())
        lines.append("")

    if report.cross_paper_insights:
        lines.append("## Cross-paper insights\n")
        for x in report.cross_paper_insights:
            lines.append(f"- {x}")
        lines.append("")

    lines.append("## Branches / subthemes\n")
    for b in report.branches_subthemes:
        lines.append(f"- **{b.name}**: {b.description}")
    if not report.branches_subthemes:
        lines.append("_None listed._")
    lines.append("")

    def _paper_section(title: str, papers: list[TopicPaperMention]) -> None:
        lines.append(f"## {title}\n")
        for p in papers:
            parts = [f"**{p.title}**"]
            if p.authors:
                parts.append("*" + ", ".join(p.authors[:6]) + "*")
            meta = []
            if p.year is not None:
                meta.append(str(p.year))
            if p.venue:
                meta.append(p.venue)
            if meta:
                parts.append("(" + "; ".join(meta) + ")")
            lines.append("- " + " ".join(parts))
            if p.arxiv_id:
                lines.append(f"  - arXiv: `{p.arxiv_id}`")
            if p.url:
                lines.append(f"  - URL: {p.url}")
            if p.note:
                lines.append(f"  - Note: {p.note}")
        if not papers:
            lines.append("_None listed._")
        lines.append("")

    _paper_section("Foundational papers (among candidates)", report.foundational_papers)
    _paper_section("Representative papers", report.representative_papers)
    _paper_section("Recent valuable papers", report.recent_valuable_papers)
    _paper_section("Lower priority / possibly overhyped (from metadata)", report.lower_priority_or_overhyped)

    lines.append("## Recent trends\n")
    for x in report.recent_trends:
        lines.append(f"- {x}")
    if not report.recent_trends:
        lines.append("_None listed._")
    lines.append("")

    lines.append("## Open questions\n")
    for x in report.open_questions:
        lines.append(f"- {x}")
    if not report.open_questions:
        lines.append("_None listed._")

    return "\n".join(lines).strip() + "\n"
