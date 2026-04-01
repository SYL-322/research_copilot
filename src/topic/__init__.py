"""Topic literature search and synthesized reports."""

from topic.literature_search import CandidatePaper, dedupe_candidates, search_literature
from topic.search import PaperHit, search_topic
from topic.topic_analyzer import TopicAnalysisError, render_topic_report_markdown
from topic.topic_service import TopicScanError, build_topic_report, topic_slug
from topic.report import generate_topic_report

__all__ = [
    "CandidatePaper",
    "PaperHit",
    "TopicAnalysisError",
    "TopicScanError",
    "build_topic_report",
    "dedupe_candidates",
    "generate_topic_report",
    "render_topic_report_markdown",
    "search_literature",
    "search_topic",
    "topic_slug",
]
