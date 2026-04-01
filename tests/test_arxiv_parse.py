"""Tests for arXiv id parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest.arxiv_client import parse_arxiv_id_from_url_or_id


class TestArxivParse(unittest.TestCase):
    def test_abs_url(self) -> None:
        u = "https://arxiv.org/abs/1706.03762"
        self.assertEqual(parse_arxiv_id_from_url_or_id(u), "1706.03762")

    def test_pdf_url(self) -> None:
        u = "https://arxiv.org/pdf/1706.03762.pdf"
        self.assertEqual(parse_arxiv_id_from_url_or_id(u), "1706.03762")

    def test_bare_id(self) -> None:
        self.assertEqual(parse_arxiv_id_from_url_or_id("arxiv:2401.12345v2"), "2401.12345v2")

    def test_old_style(self) -> None:
        self.assertEqual(parse_arxiv_id_from_url_or_id("cs.AI/0001001"), "cs.AI/0001001")


if __name__ == "__main__":
    unittest.main()
