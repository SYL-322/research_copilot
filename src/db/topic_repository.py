"""Persist versioned topic report JSON for evolving topic memory (SQLite)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.config import Settings, load_settings
from db.database import initialize_database
from utils.paths import project_root as default_project_root


def save_topic_report(
    topic: str,
    report: dict[str, Any],
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> int:
    """
    Store one snapshot of a topic report (full structured dict).

    Multiple rows per topic are allowed; newest first when reading with
    :func:`get_topic_reports`.
    """
    t = topic.strip()
    if not t:
        raise ValueError("topic is empty")
    settings = settings or load_settings()
    root = project_root if project_root is not None else default_project_root()
    payload = json.dumps(report, ensure_ascii=False)
    conn = initialize_database(settings=settings, project_root=root)
    try:
        cur = conn.execute(
            """
            INSERT INTO topic_report_versions (topic, report_json, created_at)
            VALUES (?, ?, datetime('now'))
            """,
            (t, payload),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_topic_reports(
    topic: str,
    limit: int = 3,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return the most recent stored reports for ``topic`` (newest first).

    Each element is ``{"id", "created_at", "report": {...}}``.
    """
    t = topic.strip()
    if not t:
        return []
    lim = max(1, min(int(limit), 20))
    settings = settings or load_settings()
    root = project_root if project_root is not None else default_project_root()
    conn = initialize_database(settings=settings, project_root=root)
    try:
        rows = conn.execute(
            """
            SELECT id, report_json, created_at
            FROM topic_report_versions
            WHERE topic = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (t, lim),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            parsed = json.loads(r["report_json"] or "{}")
        except json.JSONDecodeError:
            parsed = {}
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "report": parsed if isinstance(parsed, dict) else {},
            }
        )
    return out
