"""Keyword-style retrieval over stored paper memories (no vector DB)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import Settings, load_settings
from core.models import PaperMemory
from db.database import initialize_database
from db.repository import Repository
from retrieval.paper_retriever import lexical_score
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)


def _memory_search_document(title: str, mem: PaperMemory) -> str:
    """Concatenate title + structured memory JSON for lexical scoring."""
    blob = json.dumps(mem.to_memory_json(), ensure_ascii=False)
    return f"{title}\n{blob}"


def get_relevant_paper_memories(
    topic: str,
    top_k: int = 5,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Rank locally stored paper memories against ``topic`` using simple lexical overlap.

    Returns dicts suitable for JSON in the topic-scan prompt. If nothing scores above
    zero overlap, returns an empty list (metadata-only topic scan).
    """
    t = (topic or "").strip()
    if not t:
        return []

    settings = settings or load_settings()
    root = project_root if project_root is not None else default_project_root()
    k = max(1, min(int(top_k), 50))

    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        ids = repo.list_paper_ids_with_memories()
        if not ids:
            return []

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for pid in ids:
            meta = repo.get_paper_by_id(pid)
            mem = repo.get_paper_memory(pid)
            if meta is None or mem is None:
                continue
            doc = _memory_search_document(meta.title or "", mem)
            score = lexical_score(t, doc)
            payload = {
                "paper_id": pid,
                "title": meta.title or "",
                "external_id": meta.external_id,
                "source": meta.source,
                "memory": mem.to_memory_json(),
            }
            scored.append((score, pid, payload))

        if not scored:
            return []

        scored.sort(key=lambda x: (-x[0], -x[1]))
        if scored[0][0] <= 0.0:
            logger.info("Paper memory retrieval: no lexical overlap with topic; omitting memories.")
            return []

        out = [p for _, _, p in scored[:k]]
        logger.info(
            "Selected %s paper memories for topic scan (best score=%.4f)",
            len(out),
            scored[0][0],
        )
        return out
    finally:
        conn.close()
