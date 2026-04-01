#!/usr/bin/env bash
# Ingest one arXiv URL/id or local PDF path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/cli.py" ingest "$@"
