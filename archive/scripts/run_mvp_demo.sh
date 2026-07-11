#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DEMO_DIR="${DEMO_DIR:-outputs/demo}"
GENERATED_DIR="$DEMO_DIR/generated"
METRICS_DIR="$DEMO_DIR/metrics"
JOB_ID="${JOB_ID:-mvp_demo_medium_cminor}"
SEED="${SEED:-13}"

mkdir -p "$GENERATED_DIR" "$METRICS_DIR"

cat > "$DEMO_DIR/demo_request.json" <<JSON
{
  "jobId": "$JOB_ID",
  "bpm": 124,
  "timeSignature": "4/4",
  "key": "C minor",
  "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
  "bars": 2,
  "section": "drop",
  "energy": "high",
  "density": "medium",
  "style": "personal_jazz",
  "temperature": 0.9,
  "topP": 0.95,
  "modelCandidates": 2,
  "maxSequence": 256,
  "seed": $SEED
}
JSON

"$PYTHON_BIN" -m inference.app.generator \
  --bpm 124 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --bars 2 \
  --time_signature 4/4 \
  --key "C minor" \
  --section drop \
  --energy high \
  --density medium \
  --style personal_jazz \
  --temperature 0.9 \
  --top_p 0.95 \
  --model_candidates 2 \
  --max_sequence 256 \
  --seed "$SEED" \
  --job_id "$JOB_ID" \
  --output_dir "$GENERATED_DIR" \
  | tee "$DEMO_DIR/result.json"

cp "$GENERATED_DIR/$JOB_ID.mid" "$DEMO_DIR/generated.mid"
cp "$METRICS_DIR/$JOB_ID.json" "$DEMO_DIR/metrics.json"

echo "Demo MIDI: $DEMO_DIR/generated.mid"
echo "Demo metrics: $DEMO_DIR/metrics.json"
echo "Demo result: $DEMO_DIR/result.json"
