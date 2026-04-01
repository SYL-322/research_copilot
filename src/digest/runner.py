"""CLI-style entry for digest generation (delegates to :mod:`digest.digest_builder`)."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings
from digest.digest_builder import build_daily_digest


def run_digest(
    topic_slugs: list[str] | None = None,
    *,
    data_dir: Path | None = None,
    days_back: int = 3,
    settings: Settings | None = None,
    project_root: Path | None = None,
    max_per_topic: int = 15,
) -> Path:
    """
    Build a digest for the given topic strings (or all active subscriptions if empty).

    ``data_dir`` is ignored (paths come from settings); kept for API compatibility.

    Returns
    -------
    Path
        Path to the written Markdown digest file.
    """
    _ = data_dir
    topics = list(topic_slugs) if topic_slugs else None
    digest = build_daily_digest(
        topics=topics,
        days_back=days_back,
        max_per_topic=max_per_topic,
        settings=settings,
        project_root=project_root,
    )
    if not digest.digest_md_path:
        raise RuntimeError("Digest produced no markdown path.")
    return Path(digest.digest_md_path)
