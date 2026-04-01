#!/usr/bin/env bash
# Build a daily digest (optional topic args; default uses active subscriptions).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${ROOT}/cli.py" digest "$@"
