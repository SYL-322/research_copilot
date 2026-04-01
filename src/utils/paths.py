"""Filesystem paths for data layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    """Return repository root (parent of `src/`)."""
    # This file lives at src/utils/paths.py
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataPaths:
    """Standard subdirectories under the data root."""

    root: Path
    papers: Path
    topics: Path
    digests: Path


def data_paths(data_root: Path) -> DataPaths:
    """Build data path bundle; does not create directories."""
    root = data_root.resolve()
    return DataPaths(
        root=root,
        papers=root / "papers",
        topics=root / "topics",
        digests=root / "digests",
    )
