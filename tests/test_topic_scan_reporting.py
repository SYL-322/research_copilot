"""Tests for topic-scan prompt branching and report transparency."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.models import (
    Subtheme,
    TopicPaperMention,
    TopicProviderStat,
    TopicReport,
    TopicRetrievedCandidate,
    TopicRetrievalSummary,
)
from topic.literature_search import CandidatePaper, LiteratureSearchResult
from topic.topic_analyzer import build_topic_prompt, render_topic_report_markdown


class TestTopicScanReporting(unittest.TestCase):
    def test_prompt_switches_between_metadata_only_and_memory_backed(self) -> None:
        candidates = [CandidatePaper(title="Example Paper", source="arxiv", source_signals=["arxiv"])]
        metadata_prompt = build_topic_prompt(
            "4d object reconstruction / generation",
            candidates,
            project_root=ROOT,
            related_memories=[],
            retrieval_result=LiteratureSearchResult(
                topic="4d object reconstruction / generation",
                normalized_topic="4d object reconstruction / generation",
                query_variants=["4d object reconstruction / generation"],
            ),
        )
        memory_prompt = build_topic_prompt(
            "4d object reconstruction / generation",
            candidates,
            project_root=ROOT,
            related_memories=[{"title": "Example Paper", "memory": {"core_idea": "x"}}],
            retrieval_result=LiteratureSearchResult(
                topic="4d object reconstruction / generation",
                normalized_topic="4d object reconstruction / generation",
                query_variants=["4d object reconstruction / generation"],
            ),
        )

        self.assertIn("metadata_only", metadata_prompt)
        self.assertIn("Do not pretend to know method internals", metadata_prompt)
        self.assertIn("memory_backed", memory_prompt)
        self.assertIn("Use paper memories for deeper cross-paper comparison", memory_prompt)

    def test_markdown_includes_retrieval_stats_and_candidate_appendix(self) -> None:
        report = TopicReport(
            topic="4d object reconstruction / generation",
            analysis_mode="metadata_only",
            topic_summary="Summary.",
            branches_subthemes=[Subtheme(name="Geometry", description="Shape-first work.")],
            representative_papers=[
                TopicPaperMention(title="Example Paper", year=2024, note="Representative candidate.")
            ],
            evidence_quality_note="Metadata-only scan with one degraded provider.",
            retrieval_summary=TopicRetrievalSummary(
                normalized_topic="4d object reconstruction / generation",
                query_variants=["4d object reconstruction / generation", "4d object generation"],
                raw_candidates=14,
                deduped_candidates=9,
                final_candidates=6,
                report_paper_mentions=1,
                provider_stats=[
                    TopicProviderStat(
                        provider="arxiv",
                        query_variants=["4d object reconstruction / generation"],
                        raw_results=8,
                        unique_results=5,
                    ),
                    TopicProviderStat(
                        provider="semantic_scholar",
                        query_variants=["4d object generation"],
                        raw_results=6,
                        unique_results=4,
                        errors=["rate_limited"],
                        rate_limited=True,
                    ),
                ],
            ),
            retrieved_candidates=[
                TopicRetrievedCandidate(
                    title="Example Paper",
                    year=2024,
                    source="arxiv",
                    source_signals=["arxiv", "semantic_scholar"],
                    matched_queries=["4d object generation"],
                    url="https://example.com/paper",
                )
            ],
        )

        md = render_topic_report_markdown(report)

        self.assertIn("## Retrieval stats", md)
        self.assertIn("arXiv returned: 8", md)
        self.assertIn("Semantic Scholar returned: 6", md)
        self.assertIn("Deduped candidates: 9", md)
        self.assertIn("## All retrieved candidate papers", md)
        self.assertIn("Matched queries", md)


if __name__ == "__main__":
    unittest.main()
