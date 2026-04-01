"""Tests for digest Markdown rendering (no API)."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.models import DailyDigest, DailyDigestItem, DigestRecommendation
from digest.digest_builder import render_digest_markdown


class TestDigestMarkdown(unittest.TestCase):
    def test_empty_digest(self) -> None:
        d = DailyDigest(
            run_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            title="Test",
            items=[],
        )
        md = render_digest_markdown(d)
        self.assertIn("Test", md)
        self.assertIn("No items", md)

    def test_one_item(self) -> None:
        d = DailyDigest(
            run_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            title="Hot papers",
            items=[
                DailyDigestItem(
                    paper_title="A paper",
                    authors=["A", "B"],
                    recommendation=DigestRecommendation.READ,
                    why_it_matters="Because",
                    likely_limitations="Small n",
                )
            ],
        )
        md = render_digest_markdown(d)
        self.assertIn("A paper", md)
        self.assertIn("read", md.lower())


if __name__ == "__main__":
    unittest.main()
