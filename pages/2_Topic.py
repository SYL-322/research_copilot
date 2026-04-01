"""Topic mode (Streamlit UI not implemented — use `python cli.py topic` from the repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

st.set_page_config(page_title="Topic", layout="wide")
st.title("Topic mode")
st.info("CLI-first MVP: use **`python cli.py topic \"…\"`** — see README.md.")
