"""Basic logging configuration for research_copilot."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def setup_logging(
    level: str = "INFO",
    *,
    stream: TextIO | None = None,
    format_string: str | None = None,
) -> None:
    """
    Configure root logging once (safe to call multiple times).

    Parameters
    ----------
    level
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    stream
        Defaults to stderr.
    format_string
        Override default ``%(levelname)s %(name)s: %(message)s``.
    """
    stream = stream or sys.stderr
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = format_string or "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=lvl, format=fmt, stream=stream, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger (does not call setup_logging)."""
    return logging.getLogger(name)
