#!/usr/bin/env python3
"""
Initialize the SQLite database (create tables).

Run from the project root::

    python scripts/init_db.py

Or with explicit path::

    python scripts/init_db.py /path/to/research_copilot.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> int:
    root = _bootstrap_src()
    from core.config import load_settings
    from db.database import initialize_database

    parser = argparse.ArgumentParser(description="Initialize research_copilot SQLite database.")
    parser.add_argument(
        "db_path",
        nargs="?",
        default=None,
        help="Optional path to SQLite file; default from settings + project root",
    )
    args = parser.parse_args()
    settings = load_settings()
    db_path = Path(args.db_path) if args.db_path else None
    conn = initialize_database(db_path=db_path, settings=settings, project_root=root)
    try:
        out = settings.resolve_database_path(root) if db_path is None else db_path.resolve()
        print(f"Database ready: {out}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
