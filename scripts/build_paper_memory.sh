#!/usr/bin/env bash
# Build structured memory for an ingested paper (papers.id).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/cli.py" memory "$@"
