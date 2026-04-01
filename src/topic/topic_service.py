"""High-level topic report generation with search, LLM synthesis, and local cache."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings, load_settings
from core.models import TopicReport
from db.database import initialize_database
from db.repository import Repository
from db.topic_repository import get_topic_reports, save_topic_report
from retrieval.memory_retriever import get_relevant_paper_memories
from topic.literature_search import LiteratureSearchRateLimitError, search_literature
from topic.topic_analyzer import (
    TopicAnalysisError,
    llm_output_to_topic_report,
    render_topic_report_markdown,
    synthesize_topic_report,
)
from topic.topic_quality import (
    complexity_high,
    evaluate_topic_report,
    prior_claims_from_previous_reports,
    report_meets_persistence_bar,
)
from utils.files import read_text, write_text
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)

CACHE_VERSION = 5


class TopicScanError(Exception):
    """Raised when a topic report cannot be built."""


def topic_slug(topic: str) -> str:
    """Filesystem-safe slug for cache and report filenames."""
    s = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    if not s:
        return hashlib.sha256(topic.encode()).hexdigest()[:16]
    return s[:80]


def _cache_payload(report: TopicReport, topic: str, max_papers: int) -> dict[str, object]:
    return {
        "meta": {
            "version": CACHE_VERSION,
            "topic": topic,
            "max_papers": max_papers,
        },
        "report": report.model_dump(mode="json"),
    }


def _try_load_cache(
    cache_path: Path,
    topic: str,
    max_papers: int,
) -> TopicReport | None:
    if not cache_path.is_file():
        return None
    try:
        raw = json.loads(read_text(cache_path))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read topic cache %s: %s", cache_path, e)
        return None
    meta = raw.get("meta") or {}
    if meta.get("version") != CACHE_VERSION:
        return None
    if meta.get("topic") != topic or meta.get("max_papers") != max_papers:
        return None
    try:
        tr = TopicReport.model_validate(raw.get("report") or {})
    except Exception as e:
        logger.warning("Invalid cached topic report: %s", e)
        return None
    tr = tr.model_copy(
        update={
            "report_json_path": str(cache_path.resolve()),
        }
    )
    return tr


def build_topic_report(
    topic: str,
    max_papers: int = 30,
    *,
    force_refresh: bool = False,
    force_high_quality: bool = False,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> TopicReport:
    """
    Search literature, synthesize a structured report via LLM, save Markdown + JSON cache, and
    insert a row into ``topic_reports``.

    Parameters
    ----------
    topic
        Free-text research topic.
    max_papers
        Max candidates to pull from arXiv + Semantic Scholar (before dedup).
    force_refresh
        If True, skip reading cache.
    force_high_quality
        If True, always run the second (stronger-model) refinement pass.
    settings
        Defaults to :func:`core.config.load_settings`.
    project_root
        Repository root (contains ``prompts/`` and ``data/``).

    Raises
    ------
    TopicScanError
        Missing API key, empty topic, no search results, or LLM failure.
    """
    settings = settings or load_settings()
    root = project_root or default_project_root()
    t = topic.strip()
    if not t:
        raise TopicScanError("Topic is empty.")
    if not (settings.openai_api_key or "").strip():
        raise TopicScanError("OPENAI_API_KEY is not set.")

    data_dir = settings.resolve_data_dir(root)
    topics_dir = data_dir / "topics"
    cache_dir = data_dir / "cache"
    topics_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    slug = topic_slug(t)
    cache_path = cache_dir / f"topic_report_{slug}.json"
    md_path = topics_dir / f"{slug}_report.md"

    if not force_refresh and not force_high_quality:
        cached = _try_load_cache(cache_path, t, max_papers)
        if cached is not None:
            logger.info("Loaded topic report from cache: %s", cache_path)
            if cached.report_md_path and Path(cached.report_md_path).is_file():
                return cached
            # Fall through to regenerate if md missing

    try:
        candidates = search_literature(t, max_results=max_papers, settings=settings)
    except LiteratureSearchRateLimitError as e:
        raise TopicScanError(str(e)) from e
    except Exception as e:
        logger.exception("Literature search failed")
        raise TopicScanError(f"Literature search failed: {e}") from e

    if not candidates:
        raise TopicScanError(
            "No candidate papers returned. Check the topic string or try again later."
        )

    previous_reports = get_topic_reports(
        t,
        limit=3,
        settings=settings,
        project_root=root,
    )
    if previous_reports:
        logger.info("Topic scan using %s prior report snapshot(s) from DB", len(previous_reports))

    prior_claims_summary = prior_claims_from_previous_reports(previous_reports)

    related_memories = get_relevant_paper_memories(
        t,
        top_k=5,
        settings=settings,
        project_root=root,
    )
    if related_memories:
        logger.info(
            "Topic scan using %s local paper memories (lexical match)",
            len(related_memories),
        )

    model_light = settings.resolve_openai_model_light()
    model_main = settings.resolve_openai_model_main()
    logger.info("Topic scan pass 1 model=%s (light)", model_light)

    try:
        llm_initial = synthesize_topic_report(
            t,
            candidates,
            settings=settings,
            project_root=root,
            related_memories=related_memories,
            prior_claims_summary=prior_claims_summary,
            initial_topic_report_json=None,
            model=model_light,
        )
    except TopicAnalysisError as e:
        raise TopicScanError(str(e)) from e

    quality_pass1 = evaluate_topic_report(llm_initial, candidates)
    complexity = complexity_high(
        len(candidates),
        len(related_memories),
        len(previous_reports),
    )
    trigger_reasons: list[str] = []
    if quality_pass1["is_low_quality"]:
        trigger_reasons.append("low_quality")
    if complexity:
        trigger_reasons.append("complexity_high")
    if force_high_quality:
        trigger_reasons.append("force_high_quality")
    refine = bool(trigger_reasons)

    logger.info(
        "Topic scan routing: pass1_model=%s quality_score=%s complexity_high=%s "
        "second_pass=%s trigger_reasons=%s",
        model_light,
        quality_pass1["quality_score"],
        complexity,
        refine,
        trigger_reasons or ["(skip)"],
    )

    llm_final = llm_initial
    if refine:
        if model_main == model_light:
            logger.info(
                "Topic scan pass 2 model=%s (same as pass 1; refinement prompt only)",
                model_main,
            )
        else:
            logger.info("Topic scan pass 2 model=%s (main)", model_main)
        initial_json = json.dumps(
            llm_initial.model_dump(mode="json"),
            ensure_ascii=False,
        )
        try:
            llm_final = synthesize_topic_report(
                t,
                candidates,
                settings=settings,
                project_root=root,
                related_memories=related_memories,
                prior_claims_summary=prior_claims_summary,
                initial_topic_report_json=initial_json,
                model=model_main,
            )
        except TopicAnalysisError as e:
            raise TopicScanError(str(e)) from e

    report = llm_output_to_topic_report(t, llm_final, candidates)
    quality_final = evaluate_topic_report(llm_final, candidates)
    persist_ok, persist_reason = report_meets_persistence_bar(
        llm_final,
        quality_score=quality_final["quality_score"],
    )
    if not persist_ok:
        logger.warning(
            "Topic report not persisted (quality gate: %s): quality_score=%s issues=%s",
            persist_reason,
            quality_final["quality_score"],
            quality_final.get("issues", []),
        )
        return report

    logger.info(
        "Topic report passed persistence gate: quality_score=%s",
        quality_final["quality_score"],
    )
    md_body = render_topic_report_markdown(report)

    write_text(md_path, md_body)
    report = report.model_copy(
        update={
            "report_md_path": str(md_path.resolve()),
            "report_json_path": str(cache_path.resolve()),
        }
    )

    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        row_id = repo.insert_topic_report(
            t,
            report.summary,
            report.report_md_path,
            report.report_json_path,
        )
    finally:
        conn.close()

    report = report.model_copy(
        update={
            "id": row_id,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0),
        }
    )
    try:
        save_topic_report(t, report.to_full_json(), settings=settings, project_root=root)
    except Exception as e:
        logger.warning("Could not save topic memory snapshot: %s", e)
    # Refresh cache file with id + timestamps
    write_text(cache_path, json.dumps(_cache_payload(report, t, max_papers), ensure_ascii=False, indent=2) + "\n")

    logger.info("Wrote topic report id=%s md=%s json=%s", row_id, md_path, cache_path)
    return report


def run_cli() -> None:
    """``python -m topic.topic_service <topic>``."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build a topic literature report.")
    p.add_argument("topic", help="Research topic string")
    p.add_argument("--max-papers", type=int, default=30)
    p.add_argument("--force", action="store_true", help="Ignore cache")
    p.add_argument(
        "--high-quality",
        action="store_true",
        help="Always run the second (main model) refinement pass",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        r = build_topic_report(
            args.topic,
            max_papers=args.max_papers,
            force_refresh=args.force,
            force_high_quality=args.high_quality,
            project_root=root,
        )
        print(r.report_md_path)
        print(r.summary)
    except TopicScanError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    run_cli()
