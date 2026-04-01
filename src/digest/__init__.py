"""Daily hot-paper digest per subscribed topics."""

from digest.digest_builder import DigestBuildError, build_daily_digest, render_digest_markdown
from digest.recent_paper_finder import TopicPaperBatch, collect_recent_across_topics, find_recent_for_topic
from digest.runner import run_digest
from digest.subscription_service import (
    SubscriptionServiceError,
    list_subscriptions,
    pause_subscription,
    resume_subscription,
    subscribe,
    unsubscribe,
)

__all__ = [
    "DigestBuildError",
    "SubscriptionServiceError",
    "TopicPaperBatch",
    "build_daily_digest",
    "collect_recent_across_topics",
    "find_recent_for_topic",
    "list_subscriptions",
    "pause_subscription",
    "render_digest_markdown",
    "resume_subscription",
    "run_digest",
    "subscribe",
    "unsubscribe",
]
