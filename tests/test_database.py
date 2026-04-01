"""SQLite repository smoke tests (no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from core.models import PaperMemoryContent, PaperMetadata
from db.database import connect, init_schema
from db.repository import Repository


class TestDatabase(unittest.TestCase):
    def test_upsert_and_list_papers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t.db"
            conn = connect(db)
            try:
                init_schema(conn)
                repo = Repository(conn)
                pid = repo.upsert_paper(
                    PaperMetadata(
                        external_id="test:ext1",
                        title="Hello",
                        source="arxiv",
                    )
                )
                self.assertGreater(pid, 0)
                papers = repo.list_papers(limit=10)
                self.assertEqual(len(papers), 1)
                self.assertEqual(papers[0].id, pid)
                self.assertEqual(papers[0].title, "Hello")
            finally:
                conn.close()

    def test_list_stored_papers_reports_memory_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t.db"
            conn = connect(db)
            try:
                init_schema(conn)
                repo = Repository(conn)
                pid = repo.upsert_paper(
                    PaperMetadata(
                        external_id="1706.03762",
                        title="Attention Is All You Need",
                        source="arxiv",
                    )
                )
                repo.upsert_paper_memory(
                    pid,
                    json.dumps(PaperMemoryContent(title="Attention").model_dump()),
                    "test-model",
                )
                rows = repo.list_stored_papers(limit=10)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].id, pid)
                self.assertTrue(rows[0].has_memory)
            finally:
                conn.close()

    def test_search_stored_papers_matches_title_keyword_and_arxiv_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "t.db"
            conn = connect(db)
            try:
                init_schema(conn)
                repo = Repository(conn)
                repo.upsert_paper(
                    PaperMetadata(
                        external_id="1706.03762",
                        title="Attention Is All You Need",
                        authors=["Ashish Vaswani"],
                        source="arxiv",
                        venue="arXiv",
                    )
                )
                repo.upsert_paper(
                    PaperMetadata(
                        external_id="2401.00001",
                        title="Diffusion Models for Video Generation",
                        authors=["Example Author"],
                        source="arxiv",
                    )
                )

                by_title = repo.search_stored_papers("attention is all", limit=10)
                by_keyword = repo.search_stored_papers("video", limit=10)
                by_arxiv = repo.search_stored_papers("1706.03762", limit=10)

                self.assertEqual(len(by_title), 1)
                self.assertEqual(by_title[0].title, "Attention Is All You Need")
                self.assertEqual(len(by_keyword), 1)
                self.assertIn("Video", by_keyword[0].title)
                self.assertEqual(len(by_arxiv), 1)
                self.assertEqual(by_arxiv[0].external_id, "1706.03762")
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
