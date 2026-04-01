"""Tests for literature search retry and rate-limit behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.config import Settings
from topic import literature_search as ls


class _FakeClient:
    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def request(self, method: str, url: str, params=None, headers=None) -> httpx.Response:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class TestLiteratureSearch(unittest.TestCase):
    def test_expand_topic_queries_handles_slash_and_parentheses(self) -> None:
        queries = ls.expand_topic_queries("4D object reconstruction / generation (dynamic scenes)")
        self.assertIn("4D object reconstruction / generation (dynamic scenes)", queries)
        self.assertIn("4D object reconstruction / generation", queries)
        self.assertIn("4D object generation", queries)
        self.assertIn("4D object reconstruction generation", queries)

    def test_dedupe_candidates_merges_sources_and_queries(self) -> None:
        merged = ls.dedupe_candidates(
            [
                ls.CandidatePaper(
                    title="Dynamic 4D object generation",
                    source="arxiv",
                    source_signals=["arxiv"],
                    matched_queries=["4d object generation"],
                ),
                ls.CandidatePaper(
                    title="Dynamic 4D object generation",
                    source="semantic_scholar",
                    abstract="More detail",
                    source_signals=["semantic_scholar"],
                    matched_queries=["4d object reconstruction generation"],
                ),
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertCountEqual(merged[0].source_signals, ["arxiv", "semantic_scholar"])
        self.assertCountEqual(
            merged[0].matched_queries,
            ["4d object generation", "4d object reconstruction generation"],
        )

    def test_build_arxiv_query_prefers_phrase_and_conjunctive_match(self) -> None:
        query = ls._build_arxiv_query("animal dataset")

        self.assertIn('ti:"animal dataset"', query)
        self.assertIn('abs:"animal dataset"', query)
        self.assertIn("(ti:animal OR abs:animal)", query)
        self.assertIn("(ti:dataset OR abs:dataset)", query)

    def test_topic_relevance_prefers_true_phrase_match_over_generic_dataset_hit(self) -> None:
        relevant = ls.CandidatePaper(
            title="Animal Dataset Benchmark for Wildlife Recognition",
            abstract="We introduce a large animal dataset for recognition and tracking.",
            source="arxiv",
            source_signals=["arxiv"],
        )
        generic = ls.CandidatePaper(
            title="A Dataset for Document Parsing",
            abstract="This dataset targets scanned forms and OCR.",
            source="arxiv",
            source_signals=["arxiv"],
        )

        relevant_score, relevant_reasons = ls._score_topic_relevance("animal dataset", relevant)
        generic_score, generic_reasons = ls._score_topic_relevance("animal dataset", generic)

        self.assertGreater(relevant_score, generic_score)
        self.assertIn("exact_phrase:title", relevant_reasons)
        self.assertIn("generic_only_penalty", generic_reasons)

    def test_rank_and_filter_candidates_removes_generic_only_animal_dataset_hits(self) -> None:
        ranked = ls._rank_and_filter_candidates(
            "animal dataset",
            [
                ls.CandidatePaper(
                    title="A Dataset for Efficient Video Compression",
                    abstract="We collect a new benchmark for codec evaluation.",
                    year=2025,
                    source="arxiv",
                    source_signals=["arxiv"],
                    matched_queries=["animal dataset"],
                ),
                ls.CandidatePaper(
                    title="Animal Dataset Benchmark for Pose Estimation",
                    abstract="We study animal pose and release a benchmark dataset.",
                    year=2023,
                    source="semantic_scholar",
                    source_signals=["semantic_scholar"],
                    matched_queries=["animal dataset"],
                ),
                ls.CandidatePaper(
                    title="Wildlife Image Collection for Animal Re-Identification",
                    abstract="This animal dataset supports re-identification in the wild.",
                    year=2024,
                    source="arxiv",
                    source_signals=["arxiv"],
                    matched_queries=["animal dataset"],
                ),
            ],
        )

        titles = [paper.title for paper in ranked]
        self.assertNotIn("A Dataset for Efficient Video Compression", titles)
        self.assertEqual(titles[0], "Animal Dataset Benchmark for Pose Estimation")

    def test_search_literature_detailed_ranks_by_relevance_before_recency(self) -> None:
        settings = Settings(openai_api_key="x", http_timeout=1.0)
        provider_results = [
            (
                [
                    ls.CandidatePaper(
                        title="A Dataset for Efficient Video Compression",
                        abstract="A benchmark dataset for codecs.",
                        year=2026,
                        source="arxiv",
                        source_signals=["arxiv"],
                        matched_queries=["animal dataset"],
                    ),
                    ls.CandidatePaper(
                        title="Animal Dataset Benchmark for Wildlife Recognition",
                        abstract="We present an animal dataset for wildlife understanding.",
                        year=2022,
                        source="arxiv",
                        source_signals=["arxiv"],
                        matched_queries=["animal dataset"],
                    ),
                ],
                ls.ProviderQueryStat(provider="arxiv", query_variants=["animal dataset"]),
            ),
            (
                [],
                ls.ProviderQueryStat(provider="semantic_scholar", query_variants=["animal dataset"]),
            ),
        ]

        with mock.patch(
            "topic.literature_search._search_provider_variants",
            side_effect=provider_results,
        ):
            result = ls.search_literature_detailed(
                "animal dataset",
                max_results=10,
                settings=settings,
            )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(
            result.candidates[0].title,
            "Animal Dataset Benchmark for Wildlife Recognition",
        )
        self.assertGreater(result.candidates[0].topic_relevance_score, 0.0)

    def test_request_with_retries_retries_429_then_succeeds(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        rate_limited = httpx.Response(
            429,
            request=request,
            headers={"Retry-After": "0"},
        )
        ok = httpx.Response(200, request=request, text="ok")
        client = _FakeClient([rate_limited, ok])

        with mock.patch("topic.literature_search.time.sleep") as sleep_mock:
            response = ls._request_with_retries(client, "GET", "https://example.com")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.calls, 2)
        sleep_mock.assert_called_once_with(0.0)

    def test_request_with_retries_can_disable_429_retry(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        rate_limited = httpx.Response(429, request=request)
        client = _FakeClient([rate_limited])

        with self.assertRaises(httpx.HTTPStatusError):
            ls._request_with_retries(
                client,
                "GET",
                "https://example.com",
                retry_on_429=False,
            )

        self.assertEqual(client.calls, 1)

    def test_search_literature_raises_when_only_rate_limits_seen(self) -> None:
        settings = Settings(openai_api_key="x", http_timeout=1.0)

        with (
            mock.patch(
                "topic.literature_search._search_provider_variants",
                side_effect=[
                    (
                        [],
                        ls.ProviderQueryStat(
                            provider="arxiv",
                            query_variants=["4d object reconstruction"],
                            rate_limited=True,
                        ),
                    ),
                    (
                        [],
                        ls.ProviderQueryStat(
                            provider="semantic_scholar",
                            query_variants=["4d object reconstruction"],
                        ),
                    ),
                ],
            ),
        ):
            with self.assertRaises(ls.LiteratureSearchRateLimitError) as ctx:
                ls.search_literature_detailed(
                    "4d object reconstruction",
                    max_results=10,
                    settings=settings,
                )

        self.assertIn("arxiv", str(ctx.exception))

    def test_search_provider_variants_recovers_once_after_rate_limit(self) -> None:
        with (
            mock.patch(
                "topic.literature_search._search_arxiv_response",
                side_effect=[
                    ls.ProviderSearchResponse(error="rate_limited", rate_limited=True),
                    ls.ProviderSearchResponse(
                        papers=[
                            ls.CandidatePaper(
                                title="Recovered paper",
                                source="arxiv",
                                source_signals=["arxiv"],
                                matched_queries=["q2"],
                            )
                        ]
                    ),
                ],
            ),
            mock.patch("topic.literature_search.time.sleep") as sleep_mock,
        ):
            papers, stat = ls._search_provider_variants(
                "arxiv",
                ["q1", "q2"],
                max_results=10,
                timeout=1.0,
            )

        self.assertEqual(len(papers), 1)
        self.assertTrue(stat.rate_limited)
        self.assertEqual(stat.query_variants, ["q1", "q2"])
        sleep_mock.assert_called_with(6.0)

    def test_search_provider_variants_stops_after_second_rate_limit(self) -> None:
        with (
            mock.patch(
                "topic.literature_search._search_arxiv_response",
                side_effect=[
                    ls.ProviderSearchResponse(error="rate_limited", rate_limited=True),
                    ls.ProviderSearchResponse(error="rate_limited", rate_limited=True),
                    AssertionError("should stop after second rate limit"),
                ],
            ),
            mock.patch("topic.literature_search.time.sleep") as sleep_mock,
        ):
            papers, stat = ls._search_provider_variants(
                "arxiv",
                ["q1", "q2", "q3"],
                max_results=10,
                timeout=1.0,
            )

        self.assertEqual(papers, [])
        self.assertTrue(stat.rate_limited)
        sleep_mock.assert_called_with(6.0)

    def test_arxiv_search_skips_during_cooldown(self) -> None:
        ls._arxiv_cooldown_until_monotonic = 100.0
        with mock.patch("topic.literature_search.time.monotonic", return_value=95.0):
            response = ls._search_arxiv_response("3d animal", max_results=10, timeout=1.0)

        self.assertTrue(response.rate_limited)
        self.assertEqual(response.error, "rate_limited_cooldown")

    def test_semantic_scholar_throttle_waits_between_calls(self) -> None:
        ls._last_semantic_scholar_request_monotonic = None
        with (
            mock.patch("topic.literature_search.time.monotonic", side_effect=[10.0, 10.2, 11.25]),
            mock.patch("topic.literature_search.time.sleep") as sleep_mock,
        ):
            ls._wait_for_semantic_scholar_slot()
            ls._wait_for_semantic_scholar_slot()

        sleep_mock.assert_called_once()
        self.assertAlmostEqual(sleep_mock.call_args.args[0], 0.85, places=2)

    def test_arxiv_throttle_waits_between_calls(self) -> None:
        ls._last_arxiv_request_monotonic = None
        with (
            mock.patch("topic.literature_search.time.monotonic", side_effect=[10.0, 10.4, 13.7]),
            mock.patch("topic.literature_search.time.sleep") as sleep_mock,
        ):
            ls._wait_for_arxiv_slot()
            ls._wait_for_arxiv_slot()

        sleep_mock.assert_called_once()
        self.assertAlmostEqual(sleep_mock.call_args.args[0], 2.7, places=2)


if __name__ == "__main__":
    unittest.main()
