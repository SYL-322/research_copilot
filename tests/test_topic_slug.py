"""Tests for topic slug helper."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topic.topic_service import topic_slug


class TestTopicSlug(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(topic_slug("Sparse 3D Reconstruction"), "sparse_3d_reconstruction")

    def test_empty_fallback(self) -> None:
        s = topic_slug("@@@")
        self.assertEqual(len(s), 16)


if __name__ == "__main__":
    unittest.main()
