#!/usr/bin/env bash
set -euo pipefail

# Dead-air optimization sweep for Stage A.
# Phases:
#  1) Primer sweep (no retraining)
#  2) split_pitch sweep (1-epoch retraining)
#  3) Best candidate revalidation (num_samples=20 by default)
#  4) Archive artifacts

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

INPUT_DIR="./midi_dataset/midi/studio/Brad Mehldau"
BASE_LORA="./checkpoints/jazz_lora_stage_a"
BASE_COND="./data/roles/lead/000000/conditioning.mid"
BASELINE_METRICS="./samples/stage_a/metrics.json"

PRIMER_VALUES="64,96,128,160"
SPLIT_VALUES="55,60,64"

NUM_SAMPLES=10
REVAL_SAMPLES=20
EPOCHS=1
BATCH_SIZE=8
NUM_WORKERS=4
MAX_SEQUENCE=512
DEAD_AIR_THRESHOLD_MS=180
SEED=42

TRANSPOSE_ALL_KEYS="true"
OVERWRITE="true"
ARCHIVE_NAME="dead_air_sweep_artifacts.tgz"
SWEEP_ROOT="./samples/dead_air_sweep"

usage() {
  cat <<'EOF'
Run dead-air optimization sweep for Stage A.

Usage:
  bash scripts/run_dead_air_sweep.sh [options]

Options:
  --input_dir <path>                  default: ./midi_dataset/midi/studio/Brad Mehldau
  --base_lora <path>                  default: ./checkpoints/jazz_lora_stage_a
  --base_conditioning <path>          default: ./data/roles/lead/000000/conditioning.mid
  --baseline_metrics <path>           default: ./samples/stage_a/metrics.json
  --primer_values <csv>               default: 64,96,128,160
  --split_values <csv>                default: 55,60,64
  --num_samples <int>                 default: 10
  --reval_samples <int>               default: 20
  --epochs <int>                      default: 1
  --batch_size <int>                  default: 8
  --num_workers <int>                 default: 4
  --max_sequence <int>                default: 512
  --dead_air_threshold_ms <int>       default: 180
  --seed <int>                        default: 42
  --sweep_root <path>                 default: ./samples/dead_air_sweep
  --archive_name <path>               default: dead_air_sweep_artifacts.tgz
  --no_transpose_all_keys             disable transpose-all-keys in split sweep
  --no_overwrite                      disable overwrite for generated role datasets
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir) INPUT_DIR="$2"; shift 2 ;;
    --base_lora) BASE_LORA="$2"; shift 2 ;;
    --base_conditioning) BASE_COND="$2"; shift 2 ;;
    --baseline_metrics) BASELINE_METRICS="$2"; shift 2 ;;
    --primer_values) PRIMER_VALUES="$2"; shift 2 ;;
    --split_values) SPLIT_VALUES="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --reval_samples) REVAL_SAMPLES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --max_sequence) MAX_SEQUENCE="$2"; shift 2 ;;
    --dead_air_threshold_ms) DEAD_AIR_THRESHOLD_MS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sweep_root) SWEEP_ROOT="$2"; shift 2 ;;
    --archive_name) ARCHIVE_NAME="$2"; shift 2 ;;
    --no_transpose_all_keys) TRANSPOSE_ALL_KEYS="false"; shift 1 ;;
    --no_overwrite) OVERWRITE="false"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python interpreter not found (python3/python)."
    exit 1
  fi
fi

MANIFEST="${SWEEP_ROOT}/manifest.tsv"
SUMMARY_JSON="${SWEEP_ROOT}/summary.json"
SUMMARY_MD="${SWEEP_ROOT}/summary.md"

mkdir -p "$SWEEP_ROOT"
cat > "$MANIFEST" <<EOF
label	metrics_path	lora_path	conditioning_midi	primer_max_tokens	output_dir
EOF

echo "[Sweep] Base lora: $BASE_LORA"
echo "[Sweep] Base conditioning: $BASE_COND"

if [[ ! -f "${BASE_LORA}/lora_weights.pt" ]]; then
  echo "Missing lora weights: ${BASE_LORA}/lora_weights.pt"
  exit 1
fi
if [[ ! -f "${BASE_COND}" ]]; then
  echo "Missing conditioning MIDI: ${BASE_COND}"
  exit 1
fi

append_manifest() {
  local label="$1"
  local metrics_path="$2"
  local lora_path="$3"
  local conditioning="$4"
  local primer="$5"
  local output_dir="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$label" "$metrics_path" "$lora_path" "$conditioning" "$primer" "$output_dir" >> "$MANIFEST"
}

echo "[Phase 1] Primer sweep: ${PRIMER_VALUES}"
IFS=',' read -r -a PRIMERS <<< "$PRIMER_VALUES"
for P in "${PRIMERS[@]}"; do
  label="p${P}"
  out_dir="${SWEEP_ROOT}/${label}"
  metrics_path="${out_dir}/metrics.json"
  mkdir -p "$out_dir"

  "$PYTHON_BIN" scripts/generate.py \
    --lora_path "$BASE_LORA" \
    --conditioning_midi "$BASE_COND" \
    --primer_max_tokens "$P" \
    --length "$MAX_SEQUENCE" \
    --max_sequence "$MAX_SEQUENCE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --output "$out_dir"

  "$PYTHON_BIN" scripts/eval_offline_metrics.py \
    --input "$out_dir" \
    --dead_air_threshold_ms "$DEAD_AIR_THRESHOLD_MS" \
    --output_json "$metrics_path"

  append_manifest "$label" "$metrics_path" "$BASE_LORA" "$BASE_COND" "$P" "$out_dir"
done

echo "[Phase 2] split_pitch sweep: ${SPLIT_VALUES}"
IFS=',' read -r -a SPLITS <<< "$SPLIT_VALUES"
for SP in "${SPLITS[@]}"; do
  role_root="./data/roles_sp${SP}"
  lora_out="./checkpoints/jazz_lora_sp${SP}"
  label="sp${SP}_p128"
  out_dir="${SWEEP_ROOT}/${label}"
  metrics_path="${out_dir}/metrics.json"

  cmd=(
    "$PYTHON_BIN" scripts/prepare_role_dataset.py
    --input_dir "$INPUT_DIR"
    --output_dir "$role_root"
    --role lead
    --split_pitch "$SP"
    --seed "$SEED"
  )
  if [[ "$TRANSPOSE_ALL_KEYS" == "true" ]]; then
    cmd+=(--transpose_all_keys)
  fi
  if [[ "$OVERWRITE" == "true" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"

  "$PYTHON_BIN" scripts/train_qlora.py \
    --data_dir "${role_root}/lead/tokenized" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_sequence "$MAX_SEQUENCE" \
    --output_dir "$lora_out"

  cond="$(find "${role_root}/lead" -name conditioning.mid | head -n 1 || true)"
  if [[ -z "$cond" ]]; then
    echo "No conditioning.mid found under ${role_root}/lead"
    exit 1
  fi

  mkdir -p "$out_dir"
  "$PYTHON_BIN" scripts/generate.py \
    --lora_path "$lora_out" \
    --conditioning_midi "$cond" \
    --primer_max_tokens 128 \
    --length "$MAX_SEQUENCE" \
    --max_sequence "$MAX_SEQUENCE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --output "$out_dir"

  "$PYTHON_BIN" scripts/eval_offline_metrics.py \
    --input "$out_dir" \
    --dead_air_threshold_ms "$DEAD_AIR_THRESHOLD_MS" \
    --output_json "$metrics_path"

  append_manifest "$label" "$metrics_path" "$lora_out" "$cond" "128" "$out_dir"
done

echo "[Phase 3] Select best candidate"
"$PYTHON_BIN" scripts/select_best_dead_air_candidate.py \
  --manifest "$MANIFEST" \
  --baseline_metrics "$BASELINE_METRICS" \
  --output_json "$SUMMARY_JSON" \
  --output_md "$SUMMARY_MD"

BEST_LABEL=$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_candidate"]["label"])' "$SUMMARY_JSON")
BEST_LORA=$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_candidate"]["lora_path"])' "$SUMMARY_JSON")
BEST_COND=$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_candidate"]["conditioning_midi"])' "$SUMMARY_JSON")
BEST_PRIMER=$("$PYTHON_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["best_candidate"]["primer_max_tokens"])' "$SUMMARY_JSON")

echo "[Phase 4] Revalidate best candidate: ${BEST_LABEL}"
REVAL_DIR="${SWEEP_ROOT}/${BEST_LABEL}_reval${REVAL_SAMPLES}"
mkdir -p "$REVAL_DIR"

"$PYTHON_BIN" scripts/generate.py \
  --lora_path "$BEST_LORA" \
  --conditioning_midi "$BEST_COND" \
  --primer_max_tokens "$BEST_PRIMER" \
  --length "$MAX_SEQUENCE" \
  --max_sequence "$MAX_SEQUENCE" \
  --num_samples "$REVAL_SAMPLES" \
  --seed "$SEED" \
  --output "$REVAL_DIR"

"$PYTHON_BIN" scripts/eval_offline_metrics.py \
  --input "$REVAL_DIR" \
  --dead_air_threshold_ms "$DEAD_AIR_THRESHOLD_MS" \
  --output_json "${REVAL_DIR}/metrics.json"

echo "[Phase 5] Archive artifacts"
ARCHIVE_ITEMS=("samples" "checkpoints/jazz_lora_stage_a")
for SP in "${SPLITS[@]}"; do
  item="checkpoints/jazz_lora_sp${SP}"
  [[ -e "$item" ]] && ARCHIVE_ITEMS+=("$item")
done
"$PYTHON_BIN" - <<'PY' "$MANIFEST" "$SWEEP_ROOT/archive_items.txt"
import csv,sys
manifest, out = sys.argv[1], sys.argv[2]
paths = []
with open(manifest, newline='') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        paths.append(row["output_dir"])
with open(out, 'w') as f:
    for p in sorted(set(paths)):
        f.write(p + "\n")
PY
while IFS= read -r p; do
  [[ -e "$p" ]] && ARCHIVE_ITEMS+=("$p")
done < "${SWEEP_ROOT}/archive_items.txt"
[[ -e "$REVAL_DIR" ]] && ARCHIVE_ITEMS+=("$REVAL_DIR")

tar -czf "$ARCHIVE_NAME" "${ARCHIVE_ITEMS[@]}"

echo "Done."
echo "Summary JSON: $SUMMARY_JSON"
echo "Summary Markdown: $SUMMARY_MD"
echo "Best candidate: $BEST_LABEL"
echo "Revalidation metrics: ${REVAL_DIR}/metrics.json"
echo "Archive: $ARCHIVE_NAME"
