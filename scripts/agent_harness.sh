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

MODE="${1:-quick}"

usage() {
  cat <<'EOF'
Agent harness for local validation.

Usage:
  bash scripts/agent_harness.sh <mode>

Modes:
  status        Show branch and working tree status.
  quick         Run unit tests, compile checks, and git whitespace check.
  demo          Run quick checks plus the MVP demo.
  tiny-prepare  Verify tiny-overfit dataset preparation only.
  tiny-compare  Compare full-model tiny training with random-base LoRA-only.
  control-tiny  Run control_v1 full-model tiny-overfit smoke.
  stage-b-window-prepare
                Prepare Stage B phrase windows and verify token vocab fit.
  stage-b-generation-probe
                Run a one-epoch Stage B decode/generation smoke.
  stage-b-constrained-probe
                Run a constrained Stage B note-group smoke.
  stage-b-overlap-gate
                Run constrained Stage B generation with overlap postprocess gate.
  stage-b-stronger-probe
                Run a multi-sample constrained Stage B review-gate probe.
  manifest-dry-run
                Run audit -> manifest -> prepare_role_dataset smoke.
  all           Run demo and tiny-compare.

Environment:
  PYTHON_BIN    Optional Python interpreter override.
  RUN_ID        Optional run id for tiny modes.
EOF
}

print_header() {
  printf '\n== %s ==\n' "$1"
}

run_status() {
  print_header "Git status"
  git branch --show-current
  git status -sb
}

run_quick() {
  print_header "Unit tests"
  "$PYTHON_BIN" -m unittest discover tests

  print_header "Compile checks"
  "$PYTHON_BIN" -m compileall \
    scripts/generate.py \
    scripts/control_tokens.py \
    scripts/checkpoint_utils.py \
    scripts/train_qlora.py \
    scripts/run_stage_a_tiny_overfit.py \
    scripts/compare_stage_a_tiny_modes.py \
    scripts/run_control_v1_tiny_overfit.py \
    scripts/stage_b_tokens.py \
    scripts/prepare_role_dataset.py \
    scripts/run_stage_b_window_tiny_overfit.py \
    scripts/run_stage_b_generation_probe.py \
    scripts/audit_brad_mehldau_dataset.py \
    scripts/audit_jazz_piano_dataset.py \
    scripts/build_jazz_training_manifests.py \
    scripts/run_manifest_prepare_smoke.py \
    scripts/train_stage_a_full.py \
    scripts/train_stage_a_adapter.py \
    inference/app \
    tests

  print_header "Diff whitespace check"
  git diff --check
}

run_demo() {
  run_quick
  print_header "MVP demo"
  bash scripts/run_mvp_demo.sh
}

run_tiny_prepare() {
  local run_id="${RUN_ID:-harness_prepare}"
  print_header "Tiny-overfit prepare"
  "$PYTHON_BIN" scripts/run_stage_a_tiny_overfit.py \
    --prepare_only \
    --sample_count 2 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_tiny_compare() {
  local run_id="${RUN_ID:-harness_compare}"
  print_header "Stage A tiny mode comparison"
  "$PYTHON_BIN" scripts/compare_stage_a_tiny_modes.py \
    --sample_count 3 \
    --epochs 200 \
    --lr 0.001 \
    --max_sequence 128 \
    --primer_max_tokens 24 \
    --num_samples 3 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_control_tiny() {
  local run_id="${RUN_ID:-harness_control_v1}"
  print_header "Stage A control_v1 tiny overfit"
  "$PYTHON_BIN" scripts/run_control_v1_tiny_overfit.py \
    --sample_count 1 \
    --epochs 200 \
    --lr 0.001 \
    --max_sequence 192 \
    --primer_max_tokens 96 \
    --num_samples 3 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_stage_b_window_prepare() {
  local run_id="${RUN_ID:-harness_stage_b_window_prepare}"
  print_header "Stage B window prepare"
  "$PYTHON_BIN" scripts/run_stage_b_window_tiny_overfit.py \
    --run_id "$run_id" \
    --max_files 1 \
    --prepare_only
}

run_stage_b_generation_probe() {
  local run_id="${RUN_ID:-harness_stage_b_generation_probe}"
  print_header "Stage B generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 1 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 1 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_constrained_probe() {
  local run_id="${RUN_ID:-harness_stage_b_constrained_probe}"
  print_header "Stage B constrained probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 1 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 1 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --top_k 1 \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_overlap_gate() {
  local run_id="${RUN_ID:-harness_stage_b_overlap_gate}"
  print_header "Stage B overlap gate"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 1 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 1 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --top_k 1 \
    --require_note_groups \
    --require_valid_sample \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_stronger_probe() {
  local run_id="${RUN_ID:-harness_stage_b_stronger_probe}"
  print_header "Stage B stronger multi-sample probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 24 \
    --max_files 1 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --top_k 2 \
    --require_note_groups \
    --require_all_grammar_samples \
    --require_valid_sample \
    --min_valid_samples 1 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_manifest_dry_run() {
  local run_id="${RUN_ID:-harness_manifest_prepare}"
  print_header "Manifest prepare dry-run"
  "$PYTHON_BIN" scripts/run_manifest_prepare_smoke.py \
    --run_id "$run_id" \
    --audit_max_files 100 \
    --train_files 4 \
    --val_files 2 \
    --overwrite
}

case "$MODE" in
  status)
    run_status
    ;;
  quick)
    run_quick
    ;;
  demo)
    run_demo
    ;;
  tiny-prepare)
    run_tiny_prepare
    ;;
  tiny-compare)
    run_tiny_compare
    ;;
  control-tiny)
    run_control_tiny
    ;;
  stage-b-window-prepare)
    run_stage_b_window_prepare
    ;;
  stage-b-generation-probe)
    run_stage_b_generation_probe
    ;;
  stage-b-constrained-probe)
    run_stage_b_constrained_probe
    ;;
  stage-b-overlap-gate)
    run_stage_b_overlap_gate
    ;;
  stage-b-stronger-probe)
    run_stage_b_stronger_probe
    ;;
  manifest-dry-run)
    run_manifest_dry_run
    ;;
  all)
    run_demo
    run_tiny_compare
    run_control_tiny
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown harness mode: $MODE"
    usage
    exit 1
    ;;
esac
