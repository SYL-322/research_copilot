#!/usr/bin/env bash
# Ask a question about a paper (requires built memory).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/cli.py" ask "$@"
