#!/usr/bin/env bash
# Scan literature and write a topic report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/cli.py" topic "$@"
