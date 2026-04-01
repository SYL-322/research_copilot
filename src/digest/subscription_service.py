"""CRUD for digest topic subscriptions (SQLite ``subscriptions`` table)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from core.config import Settings, load_settings
from core.models import Subscription
from db.database import initialize_database
from db.repository import Repository
from topic.topic_service import topic_slug
from utils.paths import project_root as default_project_root

logger = logging.getLogger(__name__)


class SubscriptionServiceError(Exception):
    """Raised when a subscription operation fails."""


def subscribe(
    topic: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> Subscription:
    """
    Add an active subscription. ``slug`` is derived from ``topic`` (must be unique).

    Raises
    ------
    SubscriptionServiceError
        On empty topic or duplicate slug.
    """
    settings = settings or load_settings()
    root = project_root or default_project_root()
    t = topic.strip()
    if not t:
        raise SubscriptionServiceError("Topic is empty.")
    slug = topic_slug(t)
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        try:
            repo.insert_subscription(t, slug)
        except sqlite3.IntegrityError as e:
            raise SubscriptionServiceError(
                f"A subscription with slug {slug!r} already exists. Use a distinct topic wording."
            ) from e
        row = repo.get_subscription_by_slug(slug)
        if row is None:
            raise SubscriptionServiceError("Failed to read subscription after insert.")
        logger.info("Subscribed topic=%r slug=%s", t, slug)
        return row
    finally:
        conn.close()


def unsubscribe(
    slug: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> bool:
    """Remove subscription by ``slug``. Returns True if a row was deleted."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        ok = repo.delete_subscription_by_slug(slug)
        if ok:
            logger.info("Removed subscription slug=%s", slug)
        return ok
    finally:
        conn.close()


def list_subscriptions(
    *,
    active_only: bool = True,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> list[Subscription]:
    """List subscriptions."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        return repo.list_subscriptions(active_only=active_only)
    finally:
        conn.close()


def pause_subscription(
    slug: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> bool:
    """Set ``is_active`` to 0."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        return repo.set_subscription_active(slug, False)
    finally:
        conn.close()


def resume_subscription(
    slug: str,
    *,
    settings: Settings | None = None,
    project_root: Path | None = None,
) -> bool:
    """Set ``is_active`` to 1."""
    settings = settings or load_settings()
    root = project_root or default_project_root()
    conn = initialize_database(settings=settings, project_root=root)
    try:
        repo = Repository(conn)
        repo.init_schema()
        return repo.set_subscription_active(slug, True)
    finally:
        conn.close()
