"""Persistence for papers, chunks, and related entities."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from core.models import PaperChunk, PaperMemory, PaperMetadata, Subscription
from db.database import init_schema as apply_schema


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _row_to_paper_metadata(row: sqlite3.Row) -> PaperMetadata:
    authors = json.loads(row["authors_json"] or "[]")
    return PaperMetadata(
        id=int(row["id"]),
        external_id=row["external_id"],
        title=row["title"] or "",
        authors=authors if isinstance(authors, list) else [],
        year=row["year"],
        venue=row["venue"],
        source=row["source"] or "unknown",
        source_url=row["source_url"],
        pdf_path=row["pdf_path"],
        text_path=row["text_path"],
    )


class Repository:
    """SQLite persistence layer."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Underlying SQLite connection."""
        return self._conn

    def init_schema(self) -> None:
        """Create tables if missing (idempotent)."""
        apply_schema(self._conn)

    def get_paper_id_by_external_id(self, external_id: str) -> int | None:
        """Return paper id if a row exists with this ``external_id``."""
        row = self._conn.execute(
            "SELECT id FROM papers WHERE external_id = ?",
            (external_id,),
        ).fetchone()
        return int(row["id"]) if row else None

    def get_paper_by_id(self, paper_id: int) -> PaperMetadata | None:
        """Load full paper metadata by primary key."""
        row = self._conn.execute(
            "SELECT * FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_paper_metadata(row)

    def list_papers(self, *, limit: int = 200) -> list[PaperMetadata]:
        """Return papers ordered by id descending (newest first)."""
        lim = max(1, min(int(limit), 10_000))
        rows = self._conn.execute(
            "SELECT * FROM papers ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()
        return [_row_to_paper_metadata(r) for r in rows]

    def list_paper_ids_with_memories(self) -> list[int]:
        """Return ``papers.id`` for rows that have a ``paper_memories`` row (newest id first)."""
        rows = self._conn.execute(
            """
            SELECT p.id FROM papers p
            INNER JOIN paper_memories pm ON pm.paper_id = p.id
            ORDER BY p.id DESC
            """
        ).fetchall()
        return [int(r["id"]) for r in rows]

    def list_chunks_for_paper(self, paper_id: int) -> list[PaperChunk]:
        """Return chunks ordered by ``chunk_index``."""
        rows = self._conn.execute(
            """
            SELECT id, paper_id, chunk_index, content, char_start, char_end, section_title
            FROM paper_chunks WHERE paper_id = ?
            ORDER BY chunk_index ASC
            """,
            (paper_id,),
        ).fetchall()
        out: list[PaperChunk] = []
        for r in rows:
            out.append(
                PaperChunk(
                    id=int(r["id"]),
                    paper_id=int(r["paper_id"]),
                    chunk_index=int(r["chunk_index"]),
                    content=r["content"] or "",
                    char_start=r["char_start"],
                    char_end=r["char_end"],
                    section_title=r["section_title"],
                )
            )
        return out

    def upsert_paper_memory(
        self,
        paper_id: int,
        memory_json: str,
        model_used: str,
    ) -> None:
        """Insert or replace the structured memory row for ``paper_id``."""
        now = _utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO paper_memories (paper_id, memory_json, model_used, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                memory_json = excluded.memory_json,
                model_used = excluded.model_used,
                updated_at = excluded.updated_at
            """,
            (paper_id, memory_json, model_used, now, now),
        )
        self._conn.commit()

    def get_paper_memory(self, paper_id: int) -> PaperMemory | None:
        """Load structured memory from SQLite, if present."""
        row = self._conn.execute(
            "SELECT id, paper_id, memory_json, model_used, created_at, updated_at FROM paper_memories WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row["memory_json"] or "{}")
        mem = PaperMemory.from_memory_json(
            int(row["paper_id"]),
            raw,
            model_used=row["model_used"],
        )
        return mem.model_copy(
            update={
                "id": int(row["id"]),
                "created_at": _parse_iso(row["created_at"]),
                "updated_at": _parse_iso(row["updated_at"]),
            }
        )

    def upsert_paper(self, meta: PaperMetadata) -> int:
        """
        Insert or update a paper row keyed by ``external_id`` when set.

        When ``external_id`` is missing, always inserts a new row.
        """
        now = _utc_now_iso()
        authors_json = json.dumps(meta.authors, ensure_ascii=False)
        year = meta.year
        ext = meta.external_id

        if ext:
            existing = self.get_paper_id_by_external_id(ext)
            if existing is not None:
                self._conn.execute(
                    """
                    UPDATE papers SET
                        title = ?,
                        authors_json = ?,
                        year = ?,
                        venue = ?,
                        source = ?,
                        source_url = ?,
                        pdf_path = ?,
                        text_path = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        meta.title,
                        authors_json,
                        year,
                        meta.venue,
                        meta.source,
                        meta.source_url,
                        meta.pdf_path,
                        meta.text_path,
                        now,
                        existing,
                    ),
                )
                self._conn.commit()
                return existing

        cur = self._conn.execute(
            """
            INSERT INTO papers (
                external_id, title, authors_json, year, venue,
                source, source_url, pdf_path, text_path,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ext,
                meta.title,
                authors_json,
                year,
                meta.venue,
                meta.source,
                meta.source_url,
                meta.pdf_path,
                meta.text_path,
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def replace_paper_chunks(self, paper_id: int, chunks: list[PaperChunk]) -> None:
        """Delete existing chunks for ``paper_id`` and insert ``chunks`` in order."""
        self._conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
        for ch in chunks:
            self._conn.execute(
                """
                INSERT INTO paper_chunks (
                    paper_id, chunk_index, content, char_start, char_end, section_title
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    ch.chunk_index,
                    ch.content,
                    ch.char_start,
                    ch.char_end,
                    ch.section_title,
                ),
            )
        self._conn.commit()

    def save_paper_meta(self, record: dict[str, Any]) -> int:
        """Backward-compatible wrapper: build :class:`PaperMetadata` and upsert."""
        meta = PaperMetadata.model_validate(record)
        return self.upsert_paper(meta)

    def insert_topic_report(
        self,
        topic: str,
        summary: str | None,
        report_md_path: str | None,
        report_json_path: str | None,
    ) -> int:
        """Insert a topic report row; return new id."""
        now = _utc_now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO topic_reports (topic, summary, report_md_path, report_json_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (topic, summary, report_md_path, report_json_path, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def _row_to_subscription(self, row: sqlite3.Row) -> Subscription:
        return Subscription(
            id=int(row["id"]),
            topic=row["topic"] or "",
            slug=row["slug"] or "",
            is_active=bool(row["is_active"]),
            created_at=_parse_iso(row["created_at"]),
        )

    def insert_subscription(self, topic: str, slug: str) -> int:
        """Insert a subscription; raises on duplicate ``slug``."""
        now = _utc_now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO subscriptions (topic, slug, is_active, created_at)
            VALUES (?, ?, 1, ?)
            """,
            (topic.strip(), slug.strip(), now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def delete_subscription_by_slug(self, slug: str) -> bool:
        """Delete subscription row by slug. Returns True if a row was removed."""
        cur = self._conn.execute("DELETE FROM subscriptions WHERE slug = ?", (slug.strip(),))
        self._conn.commit()
        return cur.rowcount > 0

    def set_subscription_active(self, slug: str, active: bool) -> bool:
        """Update ``is_active`` by slug. Returns True if a row existed."""
        cur = self._conn.execute(
            "UPDATE subscriptions SET is_active = ? WHERE slug = ?",
            (1 if active else 0, slug.strip()),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_subscriptions(self, *, active_only: bool = True) -> list[Subscription]:
        """Return subscriptions ordered by topic."""
        if active_only:
            rows = self._conn.execute(
                "SELECT * FROM subscriptions WHERE is_active = 1 ORDER BY topic ASC"
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM subscriptions ORDER BY topic ASC").fetchall()
        return [self._row_to_subscription(r) for r in rows]

    def get_subscription_by_slug(self, slug: str) -> Subscription | None:
        row = self._conn.execute(
            "SELECT * FROM subscriptions WHERE slug = ?",
            (slug.strip(),),
        ).fetchone()
        return self._row_to_subscription(row) if row else None

    def insert_daily_digest(
        self,
        *,
        run_at: str,
        title: str,
        items_json: str,
        digest_md_path: str | None,
        subscription_id: int | None = None,
    ) -> int:
        """Insert one digest row; return id."""
        now = _utc_now_iso()
        cur = self._conn.execute(
            """
            INSERT INTO daily_digests (run_at, subscription_id, digest_md_path, title, items_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_at, subscription_id, digest_md_path, title, items_json, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)
