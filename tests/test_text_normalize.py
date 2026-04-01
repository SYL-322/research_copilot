"""Tests for title normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.text_normalize import normalize_title


class TestNormalizeTitle(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(normalize_title("  Hello   World  "), "hello world")

    def test_empty(self) -> None:
        self.assertEqual(normalize_title(""), "")


if __name__ == "__main__":
    unittest.main()
