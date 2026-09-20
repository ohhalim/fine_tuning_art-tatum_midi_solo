#!/usr/bin/env bash
# One-command demo of the continuous path.
#
#   scripts/run_continuous_demo.sh                       # no model, fallback only
#   CHECKPOINT=... PRIMER=... scripts/run_continuous_demo.sh
#   CHECKPOINT=... PRIMER=... INPUT_PORT="My Keyboard" scripts/run_continuous_demo.sh
#
# Output lands in outputs/continuous/demo: continuous_report.json and played.mid.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# MIDI ports need python-rtmidi, which the bare .venv does not carry; uv
# resolves requirements.txt for us. Set RUNNER to override.
if [[ -n "${RUNNER:-}" ]]; then
  read -r -a runner <<< "$RUNNER"
elif command -v uv >/dev/null 2>&1; then
  runner=(uv run --with-requirements requirements.txt python)
else
  runner=("${PYTHON_BIN:-python}")
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/continuous/demo}"
BARS="${BARS:-8}"
BPM="${BPM:-128}"
SEED="${SEED:-42}"

args=(--bars "$BARS" --bpm "$BPM" --seed "$SEED" --output-dir "$OUTPUT_DIR")

if [[ -n "${CHECKPOINT:-}" && -n "${PRIMER:-}" ]]; then
  args+=(--checkpoint "$CHECKPOINT" --conditioning-midi "$PRIMER")
  # Generation is measurably faster on CPU at this model size; see
  # docs/phase1/GENERATION_LATENCY.md before changing this.
  export FORCE_CPU="${FORCE_CPU:-1}"
else
  echo "No CHECKPOINT/PRIMER given: running the fallback-only path." >&2
  args+=(--fallback-only)
fi

if [[ -n "${INPUT_PORT:-}" ]]; then
  args+=(--input-port "$INPUT_PORT")
fi
if [[ -n "${OUTPUT_PORT:-}" ]]; then
  args+=(--port "$OUTPUT_PORT")
else
  # No real destination, so capture our own virtual port to prove it went out.
  args+=(--capture)
fi

"${runner[@]}" scripts/run_continuous_jazz.py "${args[@]}"
