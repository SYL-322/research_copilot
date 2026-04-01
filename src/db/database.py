"""SQLite connection, schema, and database initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from core.config import Settings

# MVP schema: foreign keys enabled; simple types; ISO8601 text timestamps.
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    authors_json TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    venue TEXT,
    source TEXT NOT NULL DEFAULT 'unknown',
    source_url TEXT,
    pdf_path TEXT,
    text_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);

CREATE TABLE IF NOT EXISTS paper_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    section_title TEXT,
    UNIQUE(paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper ON paper_chunks(paper_id);

CREATE TABLE IF NOT EXISTS paper_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    memory_json TEXT NOT NULL,
    model_used TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS topic_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    report_md_path TEXT,
    report_json_path TEXT,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_topic_reports_topic ON topic_reports(topic);

CREATE TABLE IF NOT EXISTS topic_report_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_topic_report_versions_topic ON topic_report_versions(topic);

CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    subscription_id INTEGER REFERENCES subscriptions(id) ON DELETE SET NULL,
    digest_md_path TEXT,
    title TEXT,
    items_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_daily_digests_run_at ON daily_digests(run_at);
CREATE INDEX IF NOT EXISTS idx_daily_digests_subscription ON daily_digests(subscription_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with row factory and foreign keys enforced.

    Creates parent directories for ``db_path`` if needed.
    """
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after first MVP schema (idempotent)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='paper_chunks'"
    ).fetchone()
    if not row or row["sql"] is None:
        return
    cols = conn.execute("PRAGMA table_info(paper_chunks)").fetchall()
    names = {str(r["name"]) for r in cols}
    if "section_title" not in names:
        try:
            conn.execute("ALTER TABLE paper_chunks ADD COLUMN section_title TEXT")
        except sqlite3.OperationalError:
            pass

    tr = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='topic_reports'"
    ).fetchone()
    if tr and tr["sql"]:
        tr_cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(topic_reports)").fetchall()}
        if "report_json_path" not in tr_cols:
            try:
                conn.execute("ALTER TABLE topic_reports ADD COLUMN report_json_path TEXT")
            except sqlite3.OperationalError:
                pass


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not exist."""
    conn.executescript(SCHEMA_SQL)
    _migrate_schema(conn)
    conn.commit()


def initialize_database(
    *,
    db_path: Path | None = None,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> sqlite3.Connection:
    """
    Resolve DB path, connect, and apply schema.

    Parameters
    ----------
    db_path
        Explicit file path; wins over ``settings``.
    settings
        Used with ``project_root`` to resolve ``settings.resolve_database_path``.
    project_root
        Repository root (parent of ``src/``). Required if ``db_path`` is None.

    Returns
    -------
    sqlite3.Connection
        Open connection; caller must ``close()`` when done.
    """
    if db_path is not None:
        path = db_path.expanduser().resolve()
    else:
        if settings is None:
            from core.config import load_settings

            settings = load_settings()
        if project_root is None:
            raise ValueError("project_root is required when db_path is not given")
        path = settings.resolve_database_path(project_root)

    conn = connect(path)
    try:
        init_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def database_version(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return SQLite version info (for debugging)."""
    row = conn.execute("SELECT sqlite_version() AS v").fetchone()
    return {"sqlite_version": row["v"] if row else None}
