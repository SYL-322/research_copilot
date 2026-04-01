"""Domain models for papers, memories, topics, and digests (MVP)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaperMetadata(BaseModel):
    """Bibliographic and storage metadata for one paper."""

    id: int | None = None
    external_id: str | None = Field(
        default=None,
        description="Stable id: arXiv id, DOI slug, or hash",
    )
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    source: str = Field(
        default="unknown",
        description="arxiv | semantic_scholar | upload | url | other",
    )
    source_url: str | None = None
    pdf_path: str | None = Field(default=None, description="Local path to PDF if any")
    text_path: str | None = Field(default=None, description="Local path to extracted text")
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            import json

            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            return [a.strip() for a in v.split(",") if a.strip()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


class StoredPaper(BaseModel):
    """Paper metadata with local memory availability for CLI lookup flows."""

    id: int
    external_id: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    source: str = "unknown"
    has_memory: bool = False


class GlossaryEntry(BaseModel):
    """Term and definition for paper memory."""

    model_config = ConfigDict(extra="forbid")

    term: str = ""
    definition: str = ""


class PaperMemoryContent(BaseModel):
    """
    Strict JSON shape produced by the LLM for paper memory.

    Used as the response schema for structured extraction.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    problem: str = ""
    core_idea: str = ""
    method_overview: str = ""
    key_components: list[str] = Field(default_factory=list)
    training_inference_summary: str = ""
    strongest_assumptions: list[str] = Field(default_factory=list)
    main_contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    key_insight: str | None = None
    design_rationale: str | None = None
    comparison_perspective: str | None = None
    why_it_works: str | None = None
    failure_modes: list[str] | None = None


class PaperChunk(BaseModel):
    """A text segment used for retrieval."""

    id: int | None = None
    paper_id: int = 0
    chunk_index: int = 0
    content: str = ""
    char_start: int | None = None
    char_end: int | None = None
    section_title: str | None = Field(
        default=None,
        description="Nearest inferred section heading, if any",
    )


class PaperMemory(BaseModel):
    """Structured LLM-derived memory for Q&A (stored as JSON in DB)."""

    id: int | None = None
    paper_id: int | None = None
    title: str = ""
    problem: str = ""
    core_idea: str = ""
    method_overview: str = ""
    key_components: list[str] = Field(default_factory=list)
    training_inference_summary: str = ""
    strongest_assumptions: list[str] = Field(default_factory=list)
    main_contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    key_insight: str | None = None
    design_rationale: str | None = None
    comparison_perspective: str | None = None
    why_it_works: str | None = None
    failure_modes: list[str] = Field(default_factory=list)
    truncated: bool = False
    # Legacy keys from older ingests / optional extensions
    sections: list[dict[str, Any]] = Field(default_factory=list)
    key_claims: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional structured fields")
    model_used: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("failure_modes", mode="before")
    @classmethod
    def _coerce_failure_modes(cls, v: Any) -> Any:
        if v is None:
            return []
        return v

    def to_memory_json(self) -> dict[str, Any]:
        """Serialize semantic fields for `paper_memories.memory_json` column."""
        return self.model_dump(
            mode="json",
            exclude={
                "id",
                "paper_id",
                "created_at",
                "updated_at",
                "model_used",
            },
            exclude_none=True,
        )

    @classmethod
    def from_memory_json(cls, paper_id: int, raw: dict[str, Any], **meta: Any) -> PaperMemory:
        """Build from DB JSON blob; tolerates legacy rows with missing fields."""
        data = {k: v for k, v in raw.items() if k not in {"id"}}
        data["paper_id"] = paper_id
        if meta.get("model_used") is not None:
            data["model_used"] = meta["model_used"]
        return cls.model_validate(data)

    @classmethod
    def from_llm_content(cls, content: PaperMemoryContent, *, paper_id: int, model_used: str | None) -> PaperMemory:
        """Merge strict LLM output with metadata."""
        data = content.model_dump()
        return cls(paper_id=paper_id, model_used=model_used, **data)


class Subtheme(BaseModel):
    """Named branch within a research topic."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    description: str = ""


class TopicPaperMention(BaseModel):
    """A paper referenced in a topic report (candidate or synthesized line)."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    note: str = ""
    """Why it is listed in this category (faithful to provided metadata)."""


class TopicRetrievedCandidate(BaseModel):
    """Candidate-paper metadata persisted for transparency and report appendix."""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    url: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    source: str = ""
    source_signals: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)


class TopicProviderStat(BaseModel):
    """Provider-level retrieval stats for one topic scan."""

    model_config = ConfigDict(extra="forbid")

    provider: str = ""
    query_variants: list[str] = Field(default_factory=list)
    raw_results: int = 0
    unique_results: int = 0
    errors: list[str] = Field(default_factory=list)
    rate_limited: bool = False


class TopicRetrievalSummary(BaseModel):
    """Aggregated retrieval accounting for a topic scan."""

    model_config = ConfigDict(extra="forbid")

    normalized_topic: str = ""
    query_variants: list[str] = Field(default_factory=list)
    raw_candidates: int = 0
    deduped_candidates: int = 0
    final_candidates: int = 0
    report_paper_mentions: int = 0
    provider_stats: list[TopicProviderStat] = Field(default_factory=list)


class TopicReportLlmOutput(BaseModel):
    """Strict JSON from the topic-scan LLM."""

    model_config = ConfigDict(extra="forbid")

    analysis_mode: Literal["metadata_only", "memory_backed"] = "metadata_only"
    topic_summary: str = ""
    branches_subthemes: list[Subtheme] = Field(default_factory=list)
    foundational_papers: list[TopicPaperMention] = Field(default_factory=list)
    representative_papers: list[TopicPaperMention] = Field(default_factory=list)
    recent_valuable_papers: list[TopicPaperMention] = Field(default_factory=list)
    lower_priority_or_overhyped: list[TopicPaperMention] = Field(default_factory=list)
    recent_trends: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence_quality_note: str = ""
    """How strong/limited the evidence is (API coverage, missing classics, etc.)."""
    missing_directions: list[str] = Field(
        default_factory=list,
        description="Important approaches likely missing from the candidate list",
    )
    research_frontier: str = Field(
        default="",
        description="Where the field seems to be going based on candidates and trends.",
    )
    cross_paper_insights: list[str] = Field(
        default_factory=list,
        description="Synthesis across papers using memories where available",
    )
    method_comparison_summary: str = Field(
        default="",
        description="Concise comparison of methods grounded in memories + metadata.",
    )
    evolution_notes: str = Field(
        default="",
        description="How this report updates prior topic understanding.",
    )


class TopicReport(BaseModel):
    """A generated topic overview (structured fields + file paths)."""

    id: int | None = None
    topic: str = ""
    analysis_mode: Literal["metadata_only", "memory_backed"] = "metadata_only"
    topic_summary: str = ""
    branches_subthemes: list[Subtheme] = Field(default_factory=list)
    foundational_papers: list[TopicPaperMention] = Field(default_factory=list)
    representative_papers: list[TopicPaperMention] = Field(default_factory=list)
    recent_valuable_papers: list[TopicPaperMention] = Field(default_factory=list)
    lower_priority_or_overhyped: list[TopicPaperMention] = Field(default_factory=list)
    recent_trends: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence_quality_note: str = ""
    missing_directions: list[str] = Field(default_factory=list)
    research_frontier: str = ""
    cross_paper_insights: list[str] = Field(default_factory=list)
    method_comparison_summary: str = ""
    evolution_notes: str = ""
    retrieval_summary: TopicRetrievalSummary | None = None
    retrieved_candidates: list[TopicRetrievedCandidate] = Field(default_factory=list)
    report_md_path: str | None = None
    report_json_path: str | None = None
    summary: str | None = Field(
        default=None,
        description="Short summary line for SQLite / listings",
    )
    created_at: datetime | None = None

    def to_full_json(self) -> dict[str, Any]:
        """Serialize complete report for JSON cache (excludes id/paths/timestamps optional)."""
        return self.model_dump(
            mode="json",
            exclude={"id", "report_md_path", "report_json_path", "created_at"},
            exclude_none=True,
        )


class Subscription(BaseModel):
    """User subscription to a research topic for digests."""

    id: int | None = None
    topic: str
    slug: str = ""
    is_active: bool = True
    created_at: datetime | None = None


class DigestRecommendation(str, Enum):
    """Triage label for a paper in a digest."""

    READ = "read"
    SKIM = "skim"
    IGNORE = "ignore"


class DailyDigestItem(BaseModel):
    """One paper entry inside a daily digest."""

    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    date: str | None = None
    """Human-readable publication date (ISO day or best available)."""
    source: str = ""
    """Provenance label, e.g. ``arxiv`` or ``semantic_scholar``."""
    paper_url: str | None = None
    arxiv_id: str | None = None
    matched_topics: list[str] = Field(
        default_factory=list,
        description="Subscription topics that retrieved this paper",
    )
    relevance: str = ""
    novelty: str = ""
    why_it_matters: str = ""
    likely_limitations: str = ""
    recommendation: DigestRecommendation = DigestRecommendation.SKIM
    confidence: str = ""
    time_to_invest: str = ""
    signal_strength: str = ""

    @field_validator("recommendation", mode="before")
    @classmethod
    def _coerce_recommendation(cls, v: Any) -> DigestRecommendation:
        if isinstance(v, DigestRecommendation):
            return v
        if isinstance(v, str):
            low = v.lower().strip()
            if low in ("read", "skim", "ignore"):
                return DigestRecommendation(low)
        return DigestRecommendation.SKIM


class DigestItemLlm(BaseModel):
    """Strict LLM output row for one digest paper (maps to :class:`DailyDigestItem`)."""

    model_config = ConfigDict(extra="forbid")

    paper_title: str = ""
    authors: list[str] = Field(default_factory=list)
    date: str = ""
    source: str = ""
    paper_url: str | None = None
    arxiv_id: str | None = None
    matched_topics: list[str] = Field(default_factory=list)
    relevance: str = ""
    novelty: str = ""
    likely_value: str = ""
    likely_weakness: str = ""
    why_it_matters: str = ""
    likely_limitations: str = ""
    recommendation: str = "skim"
    confidence: str = ""
    time_to_invest: str = ""
    signal_strength: str = ""


class DailyDigestLlmOutput(BaseModel):
    """Top-level LLM JSON for a digest batch."""

    model_config = ConfigDict(extra="forbid")

    items: list[DigestItemLlm] = Field(default_factory=list)


class DailyDigest(BaseModel):
    """A digest run: metadata plus items (persisted as JSON + markdown path)."""

    id: int | None = None
    run_at: datetime | None = None
    subscription_id: int | None = None
    digest_md_path: str | None = None
    digest_json_path: str | None = None
    title: str | None = None
    items: list[DailyDigestItem] = Field(default_factory=list)
    created_at: datetime | None = None

    def items_json_list(self) -> list[dict[str, Any]]:
        """Serialize items for `daily_digests.items_json`."""
        return [i.model_dump(mode="json") for i in self.items]

    @classmethod
    def from_items_json(
        cls,
        raw_items: list[dict[str, Any]] | str,
        **fields: Any,
    ) -> DailyDigest:
        """Parse DB JSON; accepts list or JSON string."""
        import json

        if isinstance(raw_items, str):
            try:
                raw_items = json.loads(raw_items)
            except json.JSONDecodeError:
                raw_items = []
        items = [DailyDigestItem.model_validate(x) for x in (raw_items or [])]
        return cls(items=items, **fields)
