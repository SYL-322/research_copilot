"""Tests for digest retrieval relevance gating and topic-source behavior."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from core.models import DailyDigestLlmOutput
from digest.digest_builder import build_daily_digest
from digest.recent_paper_finder import collect_recent_across_topics, find_recent_for_topic
from topic.literature_search import CandidatePaper, expand_topic_queries


def _paper(
    title: str,
    *,
    abstract: str = "",
    published_iso: str = "2026-03-31T00:00:00Z",
    arxiv_id: str | None = None,
    source: str = "arxiv",
) -> CandidatePaper:
    return CandidatePaper(
        title=title,
        abstract=abstract,
        published_iso=published_iso,
        arxiv_id=arxiv_id,
        source=source,
        url=f"https://example.org/{(arxiv_id or title).replace(' ', '_')}",
    )


class TestDigestRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            http_timeout=5.0,
            semantic_scholar_api_key=None,
            openai_api_key="test-key",
            resolve_openai_model_light=lambda: "test-model",
        )

    def test_broad_phrase_topic_filters_irrelevant_recent_hits(self) -> None:
        candidates = [
            _paper(
                "Animal Dataset Benchmark for Quadruped Tracking",
                abstract="A new animal dataset for wildlife tracking and behavior analysis.",
                arxiv_id="2603.00001",
            ),
            _paper(
                "Vision-Language Security Risks in Foundation Models",
                abstract="We release a benchmark dataset for generic multimodal jailbreak evaluation.",
                arxiv_id="2603.00002",
            ),
        ]
        with (
            patch("digest.recent_paper_finder.search_arxiv", return_value=candidates),
            patch("digest.recent_paper_finder.search_semantic_scholar", return_value=[]),
        ):
            out = find_recent_for_topic(
                "animal dataset",
                days_back=7,
                max_per_topic=10,
                settings=self.settings,
                fetch_cap=10,
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].title, "Animal Dataset Benchmark for Quadruped Tracking")

    def test_relevant_phrase_match_outranks_generic_keyword_hit(self) -> None:
        candidates = [
            _paper(
                "Animal Motion Synthesis with 3D Priors",
                abstract="Animal motion generation for quadrupeds from sparse supervision.",
                arxiv_id="2603.00011",
            ),
            _paper(
                "A Large Dataset for Animal Pose Forecasting",
                abstract="This dataset supports animal motion prediction benchmarks.",
                arxiv_id="2603.00012",
            ),
        ]
        with (
            patch("digest.recent_paper_finder.search_arxiv", return_value=candidates),
            patch("digest.recent_paper_finder.search_semantic_scholar", return_value=[]),
        ):
            out = find_recent_for_topic(
                "animal motion",
                days_back=7,
                max_per_topic=10,
                settings=self.settings,
                fetch_cap=10,
            )
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(out[0].title, "Animal Motion Synthesis with 3D Priors")
        self.assertGreater(out[0].topic_relevance_score, out[1].topic_relevance_score)

    def test_slash_topic_expands_into_subqueries_for_digest_retrieval(self) -> None:
        arxiv_queries: list[str] = []
        ss_queries: list[str] = []

        def fake_search_arxiv(query: str, *, max_results: int, timeout: float) -> list[CandidatePaper]:
            arxiv_queries.append(query)
            return []

        def fake_search_ss(
            query: str,
            *,
            max_results: int,
            timeout: float,
            api_key: str | None,
        ) -> list[CandidatePaper]:
            ss_queries.append(query)
            return []

        with (
            patch("digest.recent_paper_finder.search_arxiv", side_effect=fake_search_arxiv),
            patch("digest.recent_paper_finder.search_semantic_scholar", side_effect=fake_search_ss),
        ):
            out = find_recent_for_topic(
                "rigging / articulation",
                days_back=7,
                max_per_topic=10,
                settings=self.settings,
                fetch_cap=12,
            )
        self.assertEqual(out, [])
        self.assertIn("rigging / articulation", arxiv_queries)
        self.assertIn("rigging", arxiv_queries)
        self.assertIn("articulation", arxiv_queries)
        self.assertIn("articulated", arxiv_queries)
        self.assertEqual(arxiv_queries, ss_queries)

    def test_articulation_topic_keeps_articulated_reconstruction_paper(self) -> None:
        candidates = [
            _paper(
                "MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction",
                abstract="Monocular articulated 3D reconstruction with structural reasoning.",
                arxiv_id="2603.19231",
            )
        ]
        with (
            patch("digest.recent_paper_finder.search_arxiv", return_value=candidates),
            patch("digest.recent_paper_finder.search_semantic_scholar", return_value=[]),
        ):
            out = find_recent_for_topic(
                "rigging / articulation",
                days_back=60,
                max_per_topic=10,
                settings=self.settings,
                fetch_cap=12,
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].arxiv_id, "2603.19231")
        self.assertGreater(out[0].topic_relevance_score, 0.0)

    def test_expand_topic_queries_adds_close_atomic_variant_for_articulation(self) -> None:
        variants = expand_topic_queries("rigging / articulation")
        self.assertIn("articulated", variants)

    def test_collect_recent_across_topics_keeps_provenance_not_relevance_label(self) -> None:
        paper = _paper(
            "Animal Reconstruction from Multi-view Images",
            abstract="Animal reconstruction with geometry-aware priors.",
            arxiv_id="2603.00021",
        )
        with patch(
            "digest.recent_paper_finder.find_recent_for_topic",
            side_effect=[[paper], [paper], []],
        ):
            batches = collect_recent_across_topics(
                ["animal dataset", "animal reconstruction", "animal motion"],
                days_back=7,
                max_per_topic=5,
                settings=self.settings,
            )
        self.assertEqual(len(batches), 1)
        self.assertEqual(
            batches[0].matched_topics,
            ["animal dataset", "animal reconstruction"],
        )

    def test_digest_uses_subscription_topics_when_topics_not_passed(self) -> None:
        batch = SimpleNamespace(
            paper=_paper(
                "Animal Dataset Benchmark for Quadruped Tracking",
                abstract="Animal dataset paper.",
                arxiv_id="2603.00031",
            ),
            matched_topics=["animal dataset"],
        )
        fake_conn = SimpleNamespace(close=lambda: None)

        class FakeRepo:
            def __init__(self, conn: object) -> None:
                self.conn = conn

            def init_schema(self) -> None:
                return None

            def insert_daily_digest(
                self,
                *,
                run_at: str,
                title: str,
                items_json: str,
                digest_md_path: str | None,
                subscription_id: int | None,
            ) -> int:
                return 1

        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            self.settings.resolve_data_dir = lambda root: data_dir
            with (
                patch(
                    "digest.digest_builder.list_subscriptions",
                    return_value=[SimpleNamespace(topic="animal dataset")],
                ),
                patch("digest.digest_builder.collect_recent_across_topics", return_value=[batch]) as collect_mock,
                patch(
                    "digest.digest_builder._synthesize_digest",
                    return_value=DailyDigestLlmOutput(items=[]),
                ),
                patch("digest.digest_builder.initialize_database", return_value=fake_conn),
                patch("digest.digest_builder.Repository", FakeRepo),
            ):
                digest = build_daily_digest(
                    topics=None,
                    days_back=7,
                    max_per_topic=5,
                    settings=self.settings,
                    project_root=ROOT,
                )
        collect_mock.assert_called_once()
        self.assertEqual(collect_mock.call_args.args[0], ["animal dataset"])
        self.assertEqual(len(digest.items), 1)
        self.assertEqual(digest.items[0].matched_topics, ["animal dataset"])


if __name__ == "__main__":
    unittest.main()
