"""Build structured paper memory via LLM and persist to SQLite + JSON cache."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from core.config import Settings, load_settings
from core.models import PaperMemory, PaperMemoryContent
from db.database import initialize_database
from db.repository import Repository
from llm.openai_client import OpenAIClient, strip_json_fences
from utils.files import read_text, write_json
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "paper_memory_prompt.md"


class PaperMemoryBuildError(Exception):
    """Raised when memory cannot be built or persisted."""


def _resolve_paper_id(paper_id: str) -> int:
    try:
        pid = int(str(paper_id).strip())
    except ValueError as e:
        raise PaperMemoryBuildError(f"Invalid paper_id: {paper_id!r}") from e
    if pid < 1:
        raise PaperMemoryBuildError(f"paper_id must be positive, got {pid}")
    return pid


def load_prompt_template(project_root: Path) -> str:
    """Load ``prompts/paper_memory_prompt.md`` from the repository root."""
    path = project_root / "prompts" / PROMPT_FILENAME
    if not path.is_file():
        raise PaperMemoryBuildError(f"Prompt template missing: {path}")
    return path.read_text(encoding="utf-8")


def _build_paper_text_for_prompt(
    repo: Repository,
    paper_id: int,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    """Concatenate chunk texts (ordered); fall back to ``text_path`` if no chunks."""
    chunks = repo.list_chunks_for_paper(paper_id)
    if chunks:
        parts: list[str] = []
        for ch in chunks:
            label = f"[chunk {ch.chunk_index}]"
            if ch.section_title:
                label += f" [{ch.section_title}]"
            parts.append(f"{label}\n{ch.content}")
        text = "\n\n".join(parts)
    else:
        meta = repo.get_paper_by_id(paper_id)
        if meta is None or not meta.text_path:
            raise PaperMemoryBuildError(
                f"No chunks and no text_path for paper_id={paper_id}; re-ingest the paper first."
            )
        tp = Path(meta.text_path)
        if not tp.is_file():
            raise PaperMemoryBuildError(f"Text file not found: {tp}")
        text = read_text(tp)

    truncated = False
    if len(text) > max_chars:
        logger.warning("Truncating paper text from %d to %d chars", len(text), max_chars)
        text = text[:max_chars] + "\n\n[... truncated for context limit ...]"
        truncated = True
    return text, truncated


def _parse_llm_json(raw: str) -> PaperMemoryContent:
    """Validate model output; strip fences; apply light repair on failure."""
    cleaned = strip_json_fences(raw)
    try:
        return PaperMemoryContent.model_validate_json(cleaned)
    except ValidationError as e:
        logger.warning("Strict parse failed (%s); attempting repair", e)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e2:
            raise PaperMemoryBuildError(f"Model output is not valid JSON: {e2}") from e2
        _repair_memory_dict(data)
        try:
            return PaperMemoryContent.model_validate(data)
        except ValidationError as e3:
            raise PaperMemoryBuildError(
                f"Could not parse paper memory JSON after repair: {e3}"
            ) from e3


def _repair_memory_dict(data: dict[str, Any]) -> None:
    """Coerce common LLM mistakes (lists vs strings, glossary shape)."""
    list_fields = (
        "key_components",
        "strongest_assumptions",
        "main_contributions",
        "limitations",
        "open_questions",
        "failure_modes",
    )
    for k in list_fields:
        v = data.get(k)
        if isinstance(v, str):
            data[k] = [v] if v.strip() else []
        elif v is None:
            data[k] = []

    g = data.get("glossary")
    if isinstance(g, list) and g:
        fixed: list[dict[str, str]] = []
        for item in g:
            if isinstance(item, str):
                fixed.append({"term": item, "definition": ""})
            elif isinstance(item, dict):
                fixed.append(
                    {
                        "term": str(item.get("term", "")),
                        "definition": str(item.get("definition", "")),
                    }
                )
        data["glossary"] = fixed


def build_paper_memory(
    paper_id: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
    max_chars: int = 120_000,
    temperature: float = 0.1,
) -> PaperMemory:
    """
    Load ingested paper text (chunks or ``text_path``), call the LLM, validate JSON,
    store in ``paper_memories`` and ``data/cache/paper_memory_<id>.json``.

    Parameters
    ----------
    paper_id
        Primary key in ``papers`` (passed as string for CLI ergonomics).
    settings
        Defaults to :func:`core.config.load_settings`.
    project_root
        Repository root containing ``prompts/``. Defaults to :func:`utils.paths.project_root`.
    max_chars
        Maximum characters from concatenated chunks / full text sent to the model.
    temperature
        LLM sampling temperature (low for faithfulness).

    Raises
    ------
    PaperMemoryBuildError
        Missing paper, no text, invalid API key, or unparseable model output.
    """
    settings = settings or load_settings()
    root = project_root or default_project_root()
    if not (settings.openai_api_key or "").strip():
        raise PaperMemoryBuildError("OPENAI_API_KEY is not set in the environment.")

    pid = _resolve_paper_id(paper_id)
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        meta = repo.get_paper_by_id(pid)
        if meta is None:
            raise PaperMemoryBuildError(f"No paper with id={pid}")

        paper_text, truncated = _build_paper_text_for_prompt(repo, pid, max_chars=max_chars)
        if not paper_text.strip():
            raise PaperMemoryBuildError("Paper text is empty; nothing to summarize.")

        template = load_prompt_template(root)
        if "{{PAPER_TEXT}}" not in template:
            raise PaperMemoryBuildError("Prompt template must contain {{PAPER_TEXT}} placeholder.")
        user_content = template.replace("{{PAPER_TEXT}}", paper_text)

        client = OpenAIClient(settings)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant. "
                    "Follow the user instructions exactly and output only a single JSON object "
                    "with the required fields—no markdown, no preamble."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        model_used = settings.openai_model
        try:
            content = client.chat_parse(
                messages,
                PaperMemoryContent,
                model=model_used,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Primary LLM parse failed (%s); retrying with JSON object path", e)
            raw = client.chat_json_object(messages, model=model_used, temperature=temperature)
            content = _parse_llm_json(raw)

        memory = PaperMemory.from_llm_content(
            content,
            paper_id=pid,
            model_used=model_used,
        ).model_copy(update={"truncated": truncated})

        payload = memory.to_memory_json()
        repo.upsert_paper_memory(
            pid,
            json.dumps(payload, ensure_ascii=False),
            model_used,
        )

        cache_dir = settings.resolve_data_dir(root) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"paper_memory_{pid}.json"
        write_json(
            cache_path,
            {
                "paper_id": pid,
                "external_id": meta.external_id,
                "model": model_used,
                "memory": payload,
            },
        )
        logger.info("Stored paper memory paper_id=%s cache=%s", pid, cache_path)
        loaded = repo.get_paper_memory(pid)
        if loaded is None:
            raise PaperMemoryBuildError("Failed to read back paper memory from SQLite.")
        return loaded
    finally:
        conn.close()


def run_example_cli() -> None:
    """Example: ``PYTHONPATH=src python -m memory.paper_memory_builder <paper_id>``."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build structured memory for an ingested paper.")
    p.add_argument("paper_id", help="Integer papers.id from SQLite")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    try:
        mem = build_paper_memory(args.paper_id, project_root=root)
        print(json.dumps(mem.to_memory_json(), indent=2, ensure_ascii=False))
    except PaperMemoryBuildError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    run_example_cli()
