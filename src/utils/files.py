"""Read/write JSON, Markdown, and plain text with safe directory creation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_parent(path: Path) -> None:
    """Create parent directories for ``path`` if they do not exist."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file; raises FileNotFoundError if missing."""
    p = path.expanduser().resolve()
    return p.read_text(encoding=encoding)


def write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text, creating parent directories as needed."""
    p = path.expanduser().resolve()
    ensure_parent(p)
    p.write_text(content, encoding=encoding)
    logger.debug("Wrote text file: %s", p)


def read_json(path: Path, *, encoding: str = "utf-8") -> Any:
    """Load JSON from a file."""
    raw = read_text(path, encoding=encoding)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", path, e)
        raise


def write_json(
    path: Path,
    data: Any,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
) -> None:
    """Write pretty-printed JSON, creating parent directories as needed."""
    text = json.dumps(data, ensure_ascii=False, indent=indent) + "\n"
    write_text(path, text, encoding=encoding)


def read_markdown(path: Path, *, encoding: str = "utf-8") -> str:
    """Read a Markdown file as UTF-8 text (same as read_text; semantic alias)."""
    return read_text(path, encoding=encoding)


def write_markdown(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write Markdown content, creating parent directories as needed."""
    write_text(path, content, encoding=encoding)


def append_jsonl(path: Path, data: Any, *, encoding: str = "utf-8") -> None:
    """Append one JSON object as a single line, creating parent directories as needed."""
    p = path.expanduser().resolve()
    ensure_parent(p)
    with p.open("a", encoding=encoding) as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    logger.debug("Appended JSONL file: %s", p)
