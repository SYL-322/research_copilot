"""CLI argparse wiring (no network, no LLM)."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cli as cli_mod


class TestCliParser(unittest.TestCase):
    def test_ingest_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ingest", "/tmp/x.pdf"])
        self.assertEqual(args.command, "ingest")
        self.assertEqual(args.source, "/tmp/x.pdf")
        self.assertFalse(args.with_memory)

    def test_ingest_with_memory_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ingest", "--with-memory", "/tmp/x.pdf"])
        self.assertTrue(args.with_memory)

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

    def test_search_papers_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["search-papers", "attention", "--limit", "5"])
        self.assertEqual(args.command, "search-papers")
        self.assertEqual(args.query, "attention")
        self.assertEqual(args.limit, 5)


class TestCliCommands(unittest.TestCase):
    def test_cmd_ingest_outputs_next_step_guidance(self) -> None:
        result = type(
            "FakeIngestResult",
            (),
            {
                "paper_id": 7,
                "external_id": "1706.03762",
                "metadata": type("Meta", (), {"title": "Attention Is All You Need"})(),
                "chunk_count": 12,
                "text_path": ROOT / "data" / "papers" / "attention.txt",
                "cache_json_path": ROOT / "data" / "cache" / "1706.03762_ingest.json",
            },
        )()
        args = cli_mod.build_parser().parse_args(["ingest", "1706.03762"])
        out = io.StringIO()
        with (
            patch("ingest.paper_ingestor.ingest", return_value=result),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ingest(args)
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["paper_id"], 7)
        self.assertFalse(payload["memory_built"])
        self.assertIn("python cli.py memory 7", payload["next_steps"]["recommended"])
        self.assertIn("search-papers", payload["next_steps"]["find_this_paper_again"])

    def test_cmd_ingest_with_memory_builds_memory(self) -> None:
        result = type(
            "FakeIngestResult",
            (),
            {
                "paper_id": 3,
                "external_id": "2401.00001",
                "metadata": type("Meta", (), {"title": "Example Paper"})(),
                "chunk_count": 4,
                "text_path": ROOT / "data" / "papers" / "example.txt",
                "cache_json_path": ROOT / "data" / "cache" / "2401.00001_ingest.json",
            },
        )()
        memory = SimpleNamespace(title="Example Paper", truncated=False)
        args = cli_mod.build_parser().parse_args(["ingest", "--with-memory", "2401.00001"])
        out = io.StringIO()
        with (
            patch("ingest.paper_ingestor.ingest", return_value=result),
            patch("memory.paper_memory_builder.build_paper_memory", return_value=memory) as build_mem,
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ingest(args)
        self.assertEqual(rc, 0)
        build_mem.assert_called_once_with("3", project_root=ROOT)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["memory_built"])
        self.assertEqual(payload["memory"]["title"], "Example Paper")

    def test_cmd_papers_lists_rows(self) -> None:
        rows = [
            SimpleNamespace(
                id=5,
                external_id="1706.03762",
                title="Attention Is All You Need",
                year=2017,
                source="arxiv",
                has_memory=True,
            )
        ]
        args = cli_mod.build_parser().parse_args(["papers"])
        out = io.StringIO()
        fake_conn = type("FakeConn", (), {"close": lambda self: None})()
        with (
            patch("db.database.initialize_database", return_value=fake_conn),
            patch("db.repository.Repository.list_stored_papers", return_value=rows),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_papers(args)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("paper_id\tmemory\texternal_id\tyear\ttitle", text)
        self.assertIn("Attention Is All You Need", text)

    def test_cmd_search_papers_prints_matches(self) -> None:
        rows = [
            SimpleNamespace(
                id=8,
                external_id="2401.00001",
                title="Diffusion Models for Video Generation",
                year=2024,
                source="arxiv",
                has_memory=False,
            )
        ]
        args = cli_mod.build_parser().parse_args(["search-papers", "video"])
        out = io.StringIO()
        fake_conn = type("FakeConn", (), {"close": lambda self: None})()
        with (
            patch("db.database.initialize_database", return_value=fake_conn),
            patch("db.repository.Repository.search_stored_papers", return_value=rows),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_search_papers(args)
        self.assertEqual(rc, 0)
        self.assertIn("Diffusion Models for Video Generation", out.getvalue())


if __name__ == "__main__":
    unittest.main()
