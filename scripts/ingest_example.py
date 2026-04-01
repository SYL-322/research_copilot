#!/usr/bin/env python3
"""
Example: ingest one arXiv paper (or local PDF path) into SQLite and data files.

Usage (from project root)::

    PYTHONPATH=src python scripts/ingest_example.py https://arxiv.org/abs/1706.03762
    PYTHONPATH=src python scripts/ingest_example.py /path/to/paper.pdf
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from ingest.paper_ingestor import ingest

    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    result = ingest(sys.argv[1], project_root=root)
    print(
        json.dumps(
            {
                "paper_id": result.paper_id,
                "external_id": result.external_id,
                "title": result.metadata.title,
                "chunk_count": result.chunk_count,
                "text_path": str(result.text_path),
                "cache_json_path": str(result.cache_json_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
