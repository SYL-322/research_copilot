"""Smoke tests for chunking and overlap retrieval."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from retrieval.simple import chunk_text, retrieve_top_chunks


class TestRetrievalSimple(unittest.TestCase):
    def test_chunk_text_empty(self) -> None:
        self.assertEqual(chunk_text(""), [])

    def test_chunk_text_overlap(self) -> None:
        text = "a " * 100
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 1)

    def test_retrieve_top_chunks(self) -> None:
        chunks = ["hello world", "foo bar baz", "world peace"]
        top = retrieve_top_chunks("world", chunks, k=2)
        self.assertEqual(len(top), 2)
        indices = [i for i, _ in top]
        self.assertIn(0, indices)


if __name__ == "__main__":
    unittest.main()
