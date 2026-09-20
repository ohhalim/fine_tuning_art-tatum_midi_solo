#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# Resolve an interpreter that actually exists here. A bare `python` is not on
# PATH on every machine, and MIDI ports need python-rtmidi, which the plain
# .venv may not carry; uv resolves requirements.txt when it is available.
if [[ -n "${RUNNER:-}" ]]; then
  read -r -a runner <<< "$RUNNER"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  runner=("$PYTHON_BIN")
elif command -v uv >/dev/null 2>&1; then
  runner=(uv run --with-requirements requirements.txt python)
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  runner=("$ROOT_DIR/.venv/bin/python")
else
  runner=(python3)
fi
# Default demo verifies the deterministic path; model mode requires explicit paths.
if [[ $# -eq 0 ]]; then
  set -- --fallback-only --output-dir outputs/jazz_mvp/demo
fi
"${runner[@]}" scripts/run_jazz_mvp.py "$@"
