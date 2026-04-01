"""Tests for lexical paper-memory retrieval (no network)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import Settings
from core.models import PaperMemoryContent, PaperMetadata, PaperMemory
from db.database import initialize_database
from db.repository import Repository
from retrieval.memory_retriever import get_relevant_paper_memories


class TestMemoryRetriever(unittest.TestCase):
    def test_returns_empty_without_memories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            settings = Settings(
                RESEARCH_COPILOT_DATA_DIR=data,
                DATABASE_FILENAME="research_copilot.db",
            )
            conn = initialize_database(settings=settings, project_root=ROOT)
            conn.close()
            out = get_relevant_paper_memories(
                "transformer attention",
                top_k=3,
                settings=settings,
                project_root=ROOT,
            )
            self.assertEqual(out, [])

    def test_ranks_by_topic_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            settings = Settings(
                RESEARCH_COPILOT_DATA_DIR=data,
                DATABASE_FILENAME="research_copilot.db",
            )
            conn = initialize_database(settings=settings, project_root=ROOT)
            try:
                repo = Repository(conn)
                repo.init_schema()
                pid = repo.upsert_paper(
                    PaperMetadata(
                        external_id="mem:test:1",
                        title="Sparse attention mechanisms for long sequences",
                        source="arxiv",
                    )
                )
                content = PaperMemoryContent(
                    title="Sparse attention",
                    problem="We study sparse attention patterns for long context windows.",
                    core_idea="Block-sparse patterns reduce compute.",
                )
                pm = PaperMemory.from_llm_content(content, paper_id=pid, model_used="test")
                repo.upsert_paper_memory(
                    pid,
                    json.dumps(pm.to_memory_json(), ensure_ascii=False),
                    "test",
                )
            finally:
                conn.close()

            out = get_relevant_paper_memories(
                "sparse attention long context",
                top_k=2,
                settings=settings,
                project_root=ROOT,
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["paper_id"], pid)
            self.assertIn("memory", out[0])
            self.assertIn("sparse", (out[0]["memory"].get("core_idea") or "").lower())


if __name__ == "__main__":
    unittest.main()
