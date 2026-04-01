"""Tests for section chunking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.chunker import chunk_sections
from ingest.pdf_loader import SectionBlock


class TestChunker(unittest.TestCase):
    def test_plain_fallback(self) -> None:
        text = "x " * 2000
        chunks = chunk_sections([], text, chunk_size=500, overlap=50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].char_start, 0)

    def test_sections(self) -> None:
        sec = [
            SectionBlock(title="Introduction", text="word " * 800),
            SectionBlock(title="Conclusion", text="end " * 100),
        ]
        full = "\n\n".join(s.text for s in sec)
        chunks = chunk_sections(sec, full, chunk_size=400, overlap=40)
        self.assertGreater(len(chunks), 0)
        titles = {c.section_title for c in chunks}
        self.assertTrue(titles.issubset({"Introduction", "Conclusion", None}))


if __name__ == "__main__":
    unittest.main()
