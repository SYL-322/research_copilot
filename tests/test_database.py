"""SQLite repository smoke tests (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from core.models import PaperMetadata
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


if __name__ == "__main__":
    unittest.main()
