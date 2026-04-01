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

    def test_search_literature_raises_when_only_rate_limits_seen(self) -> None:
        settings = Settings(openai_api_key="x", http_timeout=1.0)

        def arxiv_side_effect(*args, **kwargs):
            ls._mark_rate_limited("arxiv")
            return []

        with (
            mock.patch("topic.literature_search.search_arxiv", side_effect=arxiv_side_effect),
            mock.patch("topic.literature_search.search_semantic_scholar", return_value=[]),
        ):
            with self.assertRaises(ls.LiteratureSearchRateLimitError) as ctx:
                ls.search_literature("4d object reconstruction", max_results=10, settings=settings)

        self.assertIn("arxiv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
