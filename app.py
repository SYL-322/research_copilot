"""
research_copilot — Streamlit entrypoint.

Adds `src/` to import path so `core`, `ingest`, etc. resolve without installing
the project as a package (MVP convenience).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from core.config import load_settings


def main() -> None:
    """Render the main app shell and navigation."""
    _ = load_settings()
    st.set_page_config(
        page_title="Research Copilot",
        page_icon="📚",
        layout="wide",
    )
    st.title("Research Copilot")
    st.caption("CLI-first MVP — use `python cli.py` from the repo root (see README).")
    st.info(
        "Primary workflow is the command line: **`python cli.py --help`**. "
        "These pages are stubs; Streamlit is optional."
    )


if __name__ == "__main__":
    main()
