"""CLI argparse wiring (no network, no LLM)."""

from __future__ import annotations

import contextlib
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
        self.assertFalse(args.save)
        self.assertEqual(args.question, ["What", "is", "this?"])

    def test_ask_save_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask", "3", "--save", "What", "is", "this?"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.paper_id, "3")
        self.assertTrue(args.save)
        self.assertEqual(args.question, ["What", "is", "this?"])

    def test_ask_save_parse_when_flag_is_after_question(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask", "3", "What", "is", "this?", "--save"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.paper_id, "3")
        self.assertTrue(args.save)
        self.assertEqual(args.question, ["What", "is", "this?"])

    def test_ask_log_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask-log", "3"])
        self.assertEqual(args.command, "ask-log")
        self.assertEqual(args.paper_id, "3")

    def test_ask_log_tail_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask-log", "3", "--tail", "2"])
        self.assertEqual(args.command, "ask-log")
        self.assertEqual(args.paper_id, "3")
        self.assertEqual(args.tail, 2)

    def test_ask_log_delete_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask-log-delete", "3"])
        self.assertEqual(args.command, "ask-log-delete")
        self.assertEqual(args.paper_id, "3")

    def test_ask_log_delete_index_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask-log-delete", "3", "--index", "2"])
        self.assertEqual(args.command, "ask-log-delete")
        self.assertEqual(args.paper_id, "3")
        self.assertEqual(args.index, 2)

    def test_ask_log_delete_question_parse(self) -> None:
        p = cli_mod.build_parser()
        args = p.parse_args(["ask-log-delete", "3", "--question", "slot attention"])
        self.assertEqual(args.command, "ask-log-delete")
        self.assertEqual(args.paper_id, "3")
        self.assertEqual(args.question, "slot attention")

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

    def test_cmd_ask_save_persists_answer(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask", "3", "--save", "What", "is", "this?"])
        out = io.StringIO()
        err = io.StringIO()
        save_path = ROOT / "data" / "papers" / "qa" / "paper_3.jsonl"
        with (
            patch("memory.paper_chat.answer_paper_question", return_value="Test answer.") as ask_mock,
            patch("memory.paper_chat.save_paper_qa_turn", return_value=save_path) as save_mock,
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            rc = cli_mod.cmd_ask(args)
        self.assertEqual(rc, 0)
        ask_mock.assert_called_once_with("3", "What is this?", project_root=ROOT)
        save_mock.assert_called_once_with("3", "What is this?", "Test answer.", project_root=ROOT)
        self.assertIn("Test answer.", out.getvalue())
        self.assertIn("Saved:", err.getvalue())

    def test_cmd_ask_save_persists_answer_when_flag_is_after_question(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask", "3", "What", "is", "this?", "--save"])
        out = io.StringIO()
        err = io.StringIO()
        save_path = ROOT / "data" / "papers" / "qa" / "paper_3.jsonl"
        with (
            patch("memory.paper_chat.answer_paper_question", return_value="Test answer.") as ask_mock,
            patch("memory.paper_chat.save_paper_qa_turn", return_value=save_path) as save_mock,
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            rc = cli_mod.cmd_ask(args)
        self.assertEqual(rc, 0)
        ask_mock.assert_called_once_with("3", "What is this?", project_root=ROOT)
        save_mock.assert_called_once_with("3", "What is this?", "Test answer.", project_root=ROOT)
        self.assertIn("Saved:", err.getvalue())

    def test_cmd_ask_log_prints_saved_turns(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log", "3"])
        out = io.StringIO()
        with (
            patch(
                "memory.paper_chat.load_paper_qa_turns",
                return_value=[
                    {
                        "timestamp": "2026-04-01T12:00:00+00:00",
                        "question": "What is this?",
                        "answer": "A saved answer.",
                    }
                ],
            ),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log(args)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("[1] 2026-04-01T12:00:00+00:00", text)
        self.assertIn("Q: What is this?", text)
        self.assertIn("A: A saved answer.", text)

    def test_cmd_ask_log_empty_history(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log", "3"])
        out = io.StringIO()
        with (
            patch("memory.paper_chat.load_paper_qa_turns", return_value=[]),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log(args)
        self.assertEqual(rc, 0)
        self.assertIn("No saved ask history for paper_id=3.", out.getvalue())

    def test_cmd_ask_log_tail_prints_recent_turns_only(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log", "3", "--tail", "2"])
        out = io.StringIO()
        with (
            patch(
                "memory.paper_chat.load_paper_qa_turns",
                return_value=[
                    {"timestamp": "2026-04-01T10:00:00+00:00", "question": "Q1", "answer": "A1"},
                    {"timestamp": "2026-04-01T11:00:00+00:00", "question": "Q2", "answer": "A2"},
                    {"timestamp": "2026-04-01T12:00:00+00:00", "question": "Q3", "answer": "A3"},
                ],
            ),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log(args)
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertNotIn("Q: Q1", text)
        self.assertIn("[2] 2026-04-01T11:00:00+00:00", text)
        self.assertIn("[3] 2026-04-01T12:00:00+00:00", text)

    def test_cmd_ask_log_delete_reports_deleted(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log-delete", "3"])
        out = io.StringIO()
        with (
            patch("memory.paper_chat.delete_paper_qa_turns", return_value=True),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log_delete(args)
        self.assertEqual(rc, 0)
        self.assertIn("Deleted ask history for paper_id=3.", out.getvalue())

    def test_cmd_ask_log_delete_reports_missing(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log-delete", "3"])
        out = io.StringIO()
        with (
            patch("memory.paper_chat.delete_paper_qa_turns", return_value=False),
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log_delete(args)
        self.assertEqual(rc, 0)
        self.assertIn("No saved ask history for paper_id=3.", out.getvalue())

    def test_cmd_ask_log_delete_by_index(self) -> None:
        args = cli_mod.build_parser().parse_args(["ask-log-delete", "3", "--index", "2"])
        out = io.StringIO()
        with (
            patch("memory.paper_chat.delete_paper_qa_turn_by_index", return_value=True) as delete_mock,
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log_delete(args)
        self.assertEqual(rc, 0)
        delete_mock.assert_called_once_with("3", 2, project_root=ROOT)
        self.assertIn("Deleted ask turn #2 for paper_id=3.", out.getvalue())

    def test_cmd_ask_log_delete_by_question(self) -> None:
        args = cli_mod.build_parser().parse_args(
            ["ask-log-delete", "3", "--question", "slot attention"]
        )
        out = io.StringIO()
        with (
            patch("memory.paper_chat.delete_paper_qa_turns_by_question", return_value=2) as delete_mock,
            patch("utils.paths.project_root", return_value=ROOT),
            redirect_stdout(out),
        ):
            rc = cli_mod.cmd_ask_log_delete(args)
        self.assertEqual(rc, 0)
        delete_mock.assert_called_once_with("3", "slot attention", project_root=ROOT)
        self.assertIn("Deleted 2 ask turn(s) for paper_id=3 matching question text.", out.getvalue())

    def test_cmd_ask_log_delete_rejects_multiple_selectors(self) -> None:
        args = cli_mod.build_parser().parse_args(
            ["ask-log-delete", "3", "--index", "2", "--question", "slot attention"]
        )
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli_mod.cmd_ask_log_delete(args)
        self.assertEqual(rc, 2)
        self.assertIn("use only one of --index or --question", err.getvalue())


if __name__ == "__main__":
    unittest.main()
