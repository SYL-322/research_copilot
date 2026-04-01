"""Tests for lexical paper chunk retrieval."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.models import PaperChunk
from retrieval.paper_retriever import lexical_score, retrieve_relevant_chunks


class TestPaperRetriever(unittest.TestCase):
    def test_lexical_score_positive(self) -> None:
        s = lexical_score("transformer attention mechanism", "we propose a transformer with attention")
        self.assertGreater(s, 0.0)

    def test_retrieve_orders_by_relevance(self) -> None:
        chunks = [
            PaperChunk(
                paper_id=1,
                chunk_index=0,
                content="The cat sat on the mat.",
            ),
            PaperChunk(
                paper_id=1,
                chunk_index=1,
                content="Attention is all you need for sequence modeling.",
            ),
        ]
        out = retrieve_relevant_chunks(chunks, "attention sequence", top_k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].chunk_index, 1)

    def test_empty_chunks(self) -> None:
        self.assertEqual(retrieve_relevant_chunks([], "hello"), [])

    def test_no_positive_score_returns_empty(self) -> None:
        chunks = [
            PaperChunk(paper_id=1, chunk_index=0, content="foo bar baz qux"),
            PaperChunk(paper_id=1, chunk_index=1, content="alpha beta gamma delta"),
        ]
        out = retrieve_relevant_chunks(chunks, "zzznonexistenttokenquery", top_k=5)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
