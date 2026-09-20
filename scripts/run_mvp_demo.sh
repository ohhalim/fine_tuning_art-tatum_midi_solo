#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x "$ROOT_DIR/.venv/bin/python" && "$PYTHON_BIN" == python ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
# Default demo verifies the deterministic path; model mode requires explicit paths.
if [[ $# -eq 0 ]]; then
  set -- --fallback-only --output-dir outputs/jazz_mvp/demo
fi
"$PYTHON_BIN" scripts/run_jazz_mvp.py "$@"
