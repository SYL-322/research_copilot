"""Topic report generation (see :mod:`topic.topic_service`)."""

from __future__ import annotations

from pathlib import Path

from core.models import TopicReport
from topic.topic_service import build_topic_report


def generate_topic_report(
    topic: str,
    hits: list[object] | None = None,
    *,
    output_path: Path | None = None,
    max_papers: int = 30,
) -> TopicReport:
    """
    Build a structured topic report (search + LLM).

    ``hits`` is ignored; the service runs its own literature search.
    ``output_path`` is ignored; paths come from :func:`build_topic_report`.
    """
    _ = (hits, output_path)
    return build_topic_report(topic, max_papers=max_papers)
