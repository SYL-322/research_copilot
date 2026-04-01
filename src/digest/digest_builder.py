"""Build a daily digest (search → LLM → SQLite + Markdown/JSON files)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings, load_settings
from core.models import DailyDigest, DailyDigestItem, DailyDigestLlmOutput, DigestItemLlm, DigestRecommendation
from db.database import initialize_database
from db.repository import Repository
from digest.recent_paper_finder import TopicPaperBatch, collect_recent_across_topics
from topic.literature_search import CandidatePaper
from utils.text_normalize import normalize_title
from digest.subscription_service import list_subscriptions
from llm.openai_client import OpenAIClient, strip_json_fences
from pydantic import ValidationError
from utils.files import write_text
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)

PROMPT_FILE = "daily_digest_prompt.md"


class DigestBuildError(Exception):
    """Raised when a digest cannot be built."""


def _load_prompt(project_root: Path) -> str:
    p = project_root / "prompts" / PROMPT_FILE
    if not p.is_file():
        raise DigestBuildError(f"Missing prompt: {p}")
    return p.read_text(encoding="utf-8")


def _batches_to_json(batches: list[TopicPaperBatch]) -> str:
    rows: list[dict[str, object]] = []
    for b in batches:
        d = b.paper.as_prompt_dict()
        d["matched_topics"] = list(b.matched_topics)
        rows.append(d)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _digest_paper_key(p: CandidatePaper) -> tuple[str, str | None]:
    aid = (p.arxiv_id or "").strip().lower() or None
    return (normalize_title(p.title), aid)


def _digest_item_key(item: DigestItemLlm) -> tuple[str, str | None]:
    aid = (item.arxiv_id or "").strip().lower() or None
    return (normalize_title(item.paper_title), aid)


def _digest_valid_keys(batches: list[TopicPaperBatch]) -> set[tuple[str, str | None]]:
    return {_digest_paper_key(b.paper) for b in batches}


def _fallback_digest_item_llm(p: CandidatePaper, matched_topics: list[str]) -> DigestItemLlm:
    return DigestItemLlm(
        paper_title=p.title,
        authors=list(p.authors[:40]),
        date=(p.published_iso or "")[:10] if p.published_iso else "",
        source=p.source,
        paper_url=p.url,
        arxiv_id=p.arxiv_id,
        matched_topics=list(matched_topics),
        relevance="",
        novelty="",
        likely_value="",
        likely_weakness="",
        why_it_matters="(Digest row not produced by LLM; metadata only.)",
        likely_limitations="",
        recommendation="skim",
        confidence="low",
        time_to_invest="medium",
        signal_strength="weak",
    )


def _align_digest_items(
    batches: list[TopicPaperBatch],
    llm_items: list[DigestItemLlm],
) -> list[DigestItemLlm]:
    """Keep only LLM rows that match a batch paper; emit one row per batch in order."""
    valid_keys = _digest_valid_keys(batches)
    filtered: list[DigestItemLlm] = []
    for item in llm_items:
        k = _digest_item_key(item)
        if k in valid_keys:
            filtered.append(item)
        else:
            logger.warning(
                "Removed digest LLM row not in input papers: %r",
                (item.paper_title or "")[:80],
            )
    by_key: dict[tuple[str, str | None], DigestItemLlm] = {}
    for item in filtered:
        k = _digest_item_key(item)
        if k not in by_key:
            by_key[k] = item
        else:
            logger.warning("Duplicate digest LLM row for key %s; keeping first", k)
    out: list[DigestItemLlm] = []
    for b in batches:
        k = _digest_paper_key(b.paper)
        if k in by_key:
            it = by_key[k]
            if not it.matched_topics:
                it = it.model_copy(update={"matched_topics": list(b.matched_topics)})
            out.append(it)
        else:
            logger.warning(
                "Missing LLM digest row for paper %r; using fallback",
                (b.paper.title or "")[:80],
            )
            out.append(_fallback_digest_item_llm(b.paper, b.matched_topics))
    return out


def _llm_item_to_digest_item(item: DigestItemLlm) -> DailyDigestItem:
    rec = item.recommendation.lower().strip()
    if rec not in ("read", "skim", "ignore"):
        rec = "skim"
    return DailyDigestItem(
        paper_title=item.paper_title,
        authors=item.authors,
        date=item.date or None,
        source=item.source,
        paper_url=item.paper_url,
        arxiv_id=item.arxiv_id,
        matched_topics=item.matched_topics,
        relevance=item.relevance,
        novelty=item.novelty,
        why_it_matters=item.why_it_matters.strip() or item.likely_value.strip(),
        likely_limitations=item.likely_limitations.strip() or item.likely_weakness.strip(),
        recommendation=DigestRecommendation(rec),
        confidence=item.confidence,
        time_to_invest=item.time_to_invest,
        signal_strength=item.signal_strength,
    )


def _synthesize_digest(
    batches: list[TopicPaperBatch],
    *,
    settings: Settings,
    project_root: Path,
    model: str | None = None,
    temperature: float = 0.2,
) -> DailyDigestLlmOutput:
    if not batches:
        return DailyDigestLlmOutput(items=[])

    template = _load_prompt(project_root)
    if "{{PAPERS_JSON}}" not in template:
        raise DigestBuildError("Prompt must contain {{PAPERS_JSON}}.")

    user_content = template.replace("{{PAPERS_JSON}}", _batches_to_json(batches))
    client = OpenAIClient(settings)
    use_model = model or settings.resolve_openai_model_light()
    logger.info("Digest synthesis model=%s", use_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You output only valid JSON with a top-level object containing an "
                '"items" array. No markdown fences. Ground every row in the provided JSON.'
            ),
        },
        {"role": "user", "content": user_content},
    ]
    try:
        # DailyDigestLlmOutput has "items" — parse as root model
        parsed = client.chat_parse(
            messages,
            DailyDigestLlmOutput,
            model=use_model,
            temperature=temperature,
        )
        return parsed
    except Exception as e:
        logger.warning("Structured parse failed (%s); retrying json_object", e)
        raw = client.chat_json_object(messages, model=use_model, temperature=temperature)
        try:
            return DailyDigestLlmOutput.model_validate_json(strip_json_fences(raw))
        except ValidationError as e2:
            raise DigestBuildError(f"Invalid digest JSON from model: {e2}") from e2


def render_digest_markdown(digest: DailyDigest) -> str:
    """Render digest as Markdown."""
    lines: list[str] = []
    title = digest.title or "Daily digest"
    lines.append(f"# {title}\n")
    rt = digest.run_at
    if isinstance(rt, datetime):
        rt_s = rt.isoformat()
    else:
        rt_s = str(rt) if rt else ""
    lines.append(f"_Generated: {rt_s}_\n")
    if not digest.items:
        lines.append("_No items._")
        return "\n".join(lines)

    for i, it in enumerate(digest.items, 1):
        lines.append(f"## {i}. {it.paper_title}\n")
        if it.authors:
            lines.append("**Authors:** " + ", ".join(it.authors[:20]) + "\n")
        meta = []
        if it.date:
            meta.append(f"**Date:** {it.date}")
        if it.source:
            meta.append(f"**Source:** {it.source}")
        if meta:
            lines.append(" | ".join(meta) + "\n")
        if it.matched_topics:
            lines.append("**Topics:** " + ", ".join(it.matched_topics) + "\n")
        if it.paper_url:
            lines.append(f"**URL:** {it.paper_url}\n")
        if it.arxiv_id:
            lines.append(f"**arXiv:** `{it.arxiv_id}`\n")
        rec = it.recommendation.value if hasattr(it.recommendation, "value") else str(it.recommendation)
        lines.append(f"**Recommendation:** `{rec}`\n")
        if it.relevance:
            lines.append(f"**Relevance:** {it.relevance}\n")
        if it.novelty:
            lines.append(f"**Novelty:** {it.novelty}\n")
        lines.append(f"**Why it matters:** {it.why_it_matters}\n")
        lines.append(f"**Limitations:** {it.likely_limitations}\n")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _digest_file_stem(topics: list[str]) -> str:
    raw = "|".join(sorted(t.strip().lower() for t in topics if t.strip()))
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"digest_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{h}"


def build_daily_digest(
    topics: list[str] | None = None,
    days_back: int = 3,
    *,
    max_per_topic: int = 15,
    model: str | None = None,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> DailyDigest:
    """
    Build a digest for the given topic strings (or all active subscriptions if ``topics`` is empty).

    Parameters
    ----------
    topics
        Search queries; if empty, uses :func:`list_subscriptions` (active only).
    days_back
        Include papers with publication date not older than this window (UTC).
    max_per_topic
        Cap per topic before cross-topic merge.

    Returns
    -------
    DailyDigest
        Persisted row + paths under ``data/digests/``.
    """
    settings = settings or load_settings()
    root = project_root or default_project_root()
    if not (settings.openai_api_key or "").strip():
        raise DigestBuildError("OPENAI_API_KEY is not set.")

    topic_list: list[str] = []
    if topics:
        topic_list = [t.strip() for t in topics if t.strip()]
    else:
        subs = list_subscriptions(active_only=True, settings=settings, project_root=root)
        topic_list = [s.topic for s in subs]

    if not topic_list:
        raise DigestBuildError("No topics: pass `topics` or add subscriptions via subscribe().")

    batches = collect_recent_across_topics(
        topic_list,
        days_back=days_back,
        max_per_topic=max_per_topic,
        settings=settings,
    )

    run_at = datetime.now(timezone.utc).replace(microsecond=0)
    title = f"Recent papers ({days_back}d): " + ", ".join(topic_list[:4])
    if len(topic_list) > 4:
        title += f" (+{len(topic_list) - 4} more)"

    if not batches:
        digest = DailyDigest(
            run_at=run_at,
            title=title + " — no papers in window",
            items=[],
            subscription_id=None,
        )
        stem = _digest_file_stem(topic_list)
        data_dir = settings.resolve_data_dir(root)
        dig_dir = data_dir / "digests"
        dig_dir.mkdir(parents=True, exist_ok=True)
        md_path = dig_dir / f"{stem}.md"
        json_path = dig_dir / f"{stem}.json"
        write_text(md_path, render_digest_markdown(digest))
        write_text(json_path, json.dumps(digest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n")
        digest = digest.model_copy(
            update={
                "digest_md_path": str(md_path.resolve()),
                "digest_json_path": str(json_path.resolve()),
            }
        )
        conn = initialize_database(settings=settings, project_root=root)
        try:
            repo = Repository(conn)
            repo.init_schema()
            rid = repo.insert_daily_digest(
                run_at=run_at.isoformat(),
                title=digest.title or "",
                items_json=json.dumps([], ensure_ascii=False),
                digest_md_path=digest.digest_md_path,
                subscription_id=None,
            )
        finally:
            conn.close()
        return digest.model_copy(update={"id": rid, "created_at": run_at})

    llm_out = _synthesize_digest(
        batches,
        settings=settings,
        project_root=root,
        model=model,
    )
    aligned = _align_digest_items(batches, llm_out.items)
    items = [_llm_item_to_digest_item(x) for x in aligned]

    digest = DailyDigest(
        run_at=run_at,
        title=title,
        items=items,
        subscription_id=None,
    )

    stem = _digest_file_stem(topic_list)
    data_dir = settings.resolve_data_dir(root)
    dig_dir = data_dir / "digests"
    dig_dir.mkdir(parents=True, exist_ok=True)
    md_path = dig_dir / f"{stem}.md"
    json_path = dig_dir / f"{stem}.json"
    write_text(md_path, render_digest_markdown(digest))
    write_text(
        json_path,
        json.dumps(digest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )
    digest = digest.model_copy(
        update={
            "digest_md_path": str(md_path.resolve()),
            "digest_json_path": str(json_path.resolve()),
        }
    )

    items_json = json.dumps(digest.items_json_list(), ensure_ascii=False)
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        rid = repo.insert_daily_digest(
            run_at=run_at.isoformat(),
            title=digest.title or "",
            items_json=items_json,
            digest_md_path=digest.digest_md_path,
            subscription_id=None,
        )
    finally:
        conn.close()

    digest = digest.model_copy(update={"id": rid, "created_at": run_at})
    logger.info("Wrote digest id=%s items=%s md=%s", rid, len(items), md_path)
    return digest


def run_cli() -> None:
    """``python -m digest.digest_builder`` with optional topic args."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build daily digest for topics.")
    p.add_argument("topics", nargs="*", help="Topics (default: all active subscriptions)")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--max-per-topic", type=int, default=15)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        td = build_daily_digest(
            topics=list(args.topics) if args.topics else None,
            days_back=args.days,
            max_per_topic=args.max_per_topic,
            project_root=root,
        )
        print(td.digest_md_path)
        print("items:", len(td.items))
    except DigestBuildError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    run_cli()
