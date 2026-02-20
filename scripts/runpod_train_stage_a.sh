#!/usr/bin/env bash
set -euo pipefail

MODE="all"
ROLE="lead"
INPUT_DIR="./midi_dataset/midi/studio/Brad Mehldau"
ROLE_ROOT="./data/roles"
CHECKPOINT_DIR="./checkpoints/jazz_lora_stage_a"
SAMPLES_DIR="./samples/stage_a"
TRAIN_EPOCHS=3
BATCH_SIZE=8
NUM_WORKERS=0
MAX_SEQUENCE=512
PRIMER_MAX_TOKENS=64
TRANSPOSE_ALL_KEYS="false"
OVERWRITE="false"
INSTALL_DEPS="false"
SEED=42
MAX_FILES=""

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

usage() {
  cat <<'EOF'
Stage A RunPod automation script.

Usage:
  bash scripts/runpod_train_stage_a.sh [options]

Options:
  --mode <prepare|train|generate|eval|all>  (default: all)
  --role <name>                              (default: lead)
  --input_dir <path>                         (default: ./midi_dataset/midi/studio/Brad Mehldau)
  --role_root <path>                         (default: ./data/roles)
  --checkpoint_dir <path>                    (default: ./checkpoints/jazz_lora_stage_a)
  --samples_dir <path>                       (default: ./samples/stage_a)
  --epochs <int>                             (default: 3)
  --batch_size <int>                         (default: 8)
  --max_sequence <int>                       (default: 512)
  --primer_max_tokens <int>                  (default: 64)
  --num_workers <int>                        (default: 0)
  --transpose_all_keys                       (flag)
  --overwrite                                (flag)
  --install_deps                             (flag)
  --seed <int>                               (default: 42)
  --max_files <int>                          (optional quick test)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --role) ROLE="$2"; shift 2 ;;
    --input_dir) INPUT_DIR="$2"; shift 2 ;;
    --role_root) ROLE_ROOT="$2"; shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --samples_dir) SAMPLES_DIR="$2"; shift 2 ;;
    --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --max_sequence) MAX_SEQUENCE="$2"; shift 2 ;;
    --primer_max_tokens) PRIMER_MAX_TOKENS="$2"; shift 2 ;;
    --transpose_all_keys) TRANSPOSE_ALL_KEYS="true"; shift 1 ;;
    --overwrite) OVERWRITE="true"; shift 1 ;;
    --install_deps) INSTALL_DEPS="true"; shift 1 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max_files) MAX_FILES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

ROLE_DIR="${ROLE_ROOT}/${ROLE}"
TOKENIZED_DIR="${ROLE_DIR}/tokenized"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python interpreter not found (python3/python)."
    exit 1
  fi
fi

run_prepare() {
  echo "[Stage A] Preparing role-conditioned dataset"
  cmd=(
    "$PYTHON_BIN" scripts/prepare_role_dataset.py
    --input_dir "$INPUT_DIR"
    --output_dir "$ROLE_ROOT"
    --role "$ROLE"
    --seed "$SEED"
  )
  if [[ "$TRANSPOSE_ALL_KEYS" == "true" ]]; then
    cmd+=(--transpose_all_keys)
  fi
  if [[ "$OVERWRITE" == "true" ]]; then
    cmd+=(--overwrite)
  fi
  if [[ -n "$MAX_FILES" ]]; then
    cmd+=(--max_files "$MAX_FILES")
  fi
  "${cmd[@]}"
}

run_train() {
  echo "[Stage A] Training LoRA model"
  "$PYTHON_BIN" scripts/train_qlora.py \
    --data_dir "$TOKENIZED_DIR" \
    --epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --max_sequence "$MAX_SEQUENCE" \
    --output_dir "$CHECKPOINT_DIR"
}

run_generate() {
  echo "[Stage A] Generating conditioned MIDI samples"
  mkdir -p "$SAMPLES_DIR"
  local conditioning_midi
  conditioning_midi="$(find "$ROLE_DIR" -mindepth 2 -maxdepth 2 -name 'conditioning.mid' | head -n 1 || true)"
  if [[ -z "$conditioning_midi" ]]; then
    echo "No conditioning.mid found under $ROLE_DIR"
    exit 1
  fi

  "$PYTHON_BIN" scripts/generate.py \
    --lora_path "$CHECKPOINT_DIR" \
    --conditioning_midi "$conditioning_midi" \
    --primer_max_tokens "$PRIMER_MAX_TOKENS" \
    --length "$MAX_SEQUENCE" \
    --max_sequence "$MAX_SEQUENCE" \
    --num_samples 5 \
    --output "$SAMPLES_DIR"
}

run_eval() {
  echo "[Stage A] Evaluating generated MIDI samples"
  "$PYTHON_BIN" scripts/eval_offline_metrics.py \
    --input "$SAMPLES_DIR" \
    --dead_air_threshold_ms 180 \
    --output_json "${SAMPLES_DIR}/metrics.json"
}

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[Setup] Installing dependencies"
  pip install -U pip
  pip install -r requirements.txt
fi

case "$MODE" in
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  generate)
    run_generate
    ;;
  eval)
    run_eval
    ;;
  all)
    run_prepare
    run_train
    run_generate
    run_eval
    ;;
  *)
    echo "Invalid mode: $MODE"
    usage
    exit 1
    ;;
esac

echo "Done: mode=$MODE role=$ROLE"
