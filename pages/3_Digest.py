"""Digest (Streamlit UI not implemented — use `python cli.py digest` from the repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

st.set_page_config(page_title="Digest", layout="wide")
st.title("Daily digest")
st.info(
    "CLI-first MVP: use **`python cli.py digest`**, **`subscribe`**, **`subscriptions`** — see README.md."
)
