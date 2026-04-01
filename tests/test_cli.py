"""CLI argparse wiring (no network, no LLM)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cli as cli_mod


class TestCliParser(unittest.TestCase):
    def test_ingest_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ingest", "/tmp/x.pdf"])
        self.assertEqual(args.command, "ingest")
        self.assertEqual(args.source, "/tmp/x.pdf")

    def test_ask_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask", "3", "What", "is", "this?"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.paper_id, "3")
        self.assertEqual(args.question, ["What", "is", "this?"])

    def test_digest_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["digest", "--days", "7", "topic", "a"])
        self.assertEqual(args.command, "digest")
        self.assertEqual(args.days, 7)
        self.assertEqual(args.topics, ["topic", "a"])

    def test_topic_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(
            ["topic", "diffusion-models", "--force", "--max-papers", "10", "--high-quality"]
        )
        self.assertEqual(args.command, "topic")
        self.assertEqual(args.topic, "diffusion-models")
        self.assertTrue(args.force)
        self.assertEqual(args.max_papers, 10)
        self.assertTrue(args.high_quality)


if __name__ == "__main__":
    unittest.main()
