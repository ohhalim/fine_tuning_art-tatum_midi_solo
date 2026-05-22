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
  stage-b-collapse-sweep
                Run Stage B top-k sampling sweep with collapse diagnostics.
  stage-b-2file-brad-probe
                Run a two-file Brad Stage B generation probe with strict reporting.
  stage-b-coverage-aware-probe
                Run a two-file Brad Stage B coverage-aware generation probe.
  stage-b-coverage-ab-sweep
                Run plain vs coverage-aware Stage B constrained generation sweep.
  stage-b-candidate-ranking
                Run coverage A/B sweep and rank generated MIDI candidates.
  stage-b-chord-aware-probe
                Run coverage/chord-aware Stage B sweep and rank candidates.
  stage-b-longer-phrase-probe
                Run a 4-bar coverage/chord-aware Stage B probe and export review MIDI.
  stage-b-pitch-mode-compare
                Compare 4-bar tones vs tones_tensions Stage B pitch modes.
  stage-b-8bar-approach-phrase
                Compare 8-bar tones/tensions/approach Stage B phrase candidates.
  stage-b-swing-motif-phrase
                Compare 8-bar baseline approach with swing/motif phrase grammar.
  stage-b-reference-stats
                Build real Stage B phrase-window reference statistics.
  stage-b-motif-templates
                Extract data-derived Stage B motif/rhythm templates.
  stage-b-data-motif-compare
                Compare hand-written swing against data-derived motif baseline.
  stage-b-data-motif-review-export
                Export named MIDI review candidates for data-derived motif compare.
  stage-b-review-context-grid
                Export review MIDI with chord context and straight-grid reference.
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
    scripts/run_stage_b_sampling_sweep.py \
    scripts/run_stage_b_coverage_ab_sweep.py \
    scripts/run_stage_b_pitch_mode_compare.py \
    scripts/run_stage_b_phrase_grammar_compare.py \
    scripts/run_stage_b_reference_stats.py \
    scripts/run_stage_b_motif_template_extraction.py \
    scripts/run_stage_b_data_motif_generation_compare.py \
    scripts/rank_stage_b_candidates.py \
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

run_stage_b_collapse_sweep() {
  local run_id="${RUN_ID:-harness_stage_b_collapse_sweep}"
  print_header "Stage B collapse sampling sweep"
  "$PYTHON_BIN" scripts/run_stage_b_sampling_sweep.py \
    --run_id "$run_id" \
    --issue_number 31 \
    --top_ks 1,2 \
    --temperatures 0.9 \
    --train_top_k 2 \
    --max_files 1 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --constrained_note_groups_per_bar 4 \
    --max_simultaneous_notes 2 \
    --require_all_grammar_samples \
    --min_best_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_2file_brad_probe() {
  local run_id="${RUN_ID:-harness_stage_b_2file_brad_probe}"
  print_header "Stage B 2-file Brad generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 33 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_coverage_aware_probe() {
  local run_id="${RUN_ID:-harness_stage_b_coverage_aware_probe}"
  print_header "Stage B coverage-aware generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 37 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --coverage_aware_positions \
    --coverage_position_window 0 \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_coverage_ab_sweep() {
  local run_id="${RUN_ID:-harness_stage_b_coverage_ab_sweep}"
  print_header "Stage B coverage-aware A/B sweep"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$run_id" \
    --issue_number 39 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_candidate_ranking() {
  local run_id="${RUN_ID:-harness_stage_b_candidate_ranking}"
  local sweep_run_id="${run_id}_ab_sweep"
  print_header "Stage B candidate ranking"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$sweep_run_id" \
    --issue_number 43 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/rank_stage_b_candidates.py \
    --run_id "$run_id" \
    --issue_number 43 \
    --ab_sweep_report "outputs/stage_b_coverage_ab_sweep/${sweep_run_id}/ab_sweep_report.json" \
    --top_n 12 \
    --min_top_strict_candidates 1
}

run_stage_b_chord_aware_probe() {
  local run_id="${RUN_ID:-harness_stage_b_chord_aware_probe}"
  local sweep_run_id="${run_id}_ab_sweep"
  print_header "Stage B chord-aware pitch probe"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$sweep_run_id" \
    --issue_number 45 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage,coverage_chord \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --chord_pitch_mode tones \
    --chord_pitch_repeat_window 2 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/rank_stage_b_candidates.py \
    --run_id "$run_id" \
    --issue_number 45 \
    --ab_sweep_report "outputs/stage_b_coverage_ab_sweep/${sweep_run_id}/ab_sweep_report.json" \
    --top_n 12 \
    --min_top_strict_candidates 1
}

run_stage_b_longer_phrase_probe() {
  local run_id="${RUN_ID:-harness_stage_b_longer_phrase_probe}"
  print_header "Stage B longer coverage/chord-aware phrase probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 49 \
    --max_files 2 \
    --window_bars 4 \
    --window_stride_bars 2 \
    --min_window_target_notes 8 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 192 \
    --num_samples 3 \
    --bars 4 \
    --generation_mode constrained \
    --coverage_aware_positions \
    --coverage_position_window 0 \
    --chord_aware_pitches \
    --chord_pitch_mode tones \
    --chord_pitch_repeat_window 2 \
    --constrained_note_groups_per_bar 8 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/export_stage_b_review_candidates.py \
    --source_report "outputs/stage_b_generation_probe/${run_id}/report.json" \
    --run_id "$run_id" \
    --top_n 3 \
    --mode coverage_chord \
    --copy_midi
}

run_stage_b_pitch_mode_compare() {
  local run_id="${RUN_ID:-harness_stage_b_pitch_mode_compare}"
  print_header "Stage B tones vs tones_tensions pitch-mode comparison"
  "$PYTHON_BIN" scripts/run_stage_b_pitch_mode_compare.py \
    --run_id "$run_id" \
    --issue_number 55 \
    --max_files 2 \
    --window_bars 4 \
    --window_stride_bars 2 \
    --min_window_target_notes 8 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 192 \
    --num_samples 3 \
    --bars 4 \
    --pitch_modes tones,tones_tensions \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_8bar_approach_phrase() {
  local run_id="${RUN_ID:-harness_stage_b_8bar_approach_phrase}"
  print_header "Stage B 8-bar approach phrase comparison"
  "$PYTHON_BIN" scripts/run_stage_b_pitch_mode_compare.py \
    --run_id "$run_id" \
    --issue_number 57 \
    --max_files 2 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 384 \
    --num_samples 3 \
    --bars 8 \
    --pitch_modes tones,tones_tensions,approach_tensions \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_swing_motif_phrase() {
  local run_id="${RUN_ID:-harness_stage_b_swing_motif_phrase}"
  print_header "Stage B swing/motif phrase grammar comparison"
  "$PYTHON_BIN" scripts/run_stage_b_phrase_grammar_compare.py \
    --run_id "$run_id" \
    --issue_number 59 \
    --max_files 2 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 384 \
    --num_samples 3 \
    --bars 8 \
    --grammar_modes approach_baseline,swing_motif_approach \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_reference_stats() {
  local run_id="${RUN_ID:-harness_stage_b_reference_stats}"
  print_header "Stage B reference phrase statistics"
  "$PYTHON_BIN" scripts/run_stage_b_reference_stats.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --max_records 64 \
    --generated_report outputs/stage_b_phrase_grammar_compare/harness_stage_b_swing_motif_phrase/phrase_grammar_compare_report.json
}

run_stage_b_motif_templates() {
  local run_id="${RUN_ID:-harness_stage_b_motif_templates}"
  print_header "Stage B motif template extraction"
  "$PYTHON_BIN" scripts/run_stage_b_motif_template_extraction.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --top_n 20
}

run_stage_b_data_motif_compare() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_compare}"
  print_header "Stage B data-derived motif generation compare"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8
}

run_stage_b_data_motif_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_review_export}"
  print_header "Stage B data motif review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
}

run_stage_b_review_context_grid() {
  local run_id="${RUN_ID:-harness_stage_b_review_context_grid}"
  print_header "Stage B review context and straight-grid export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,hand_written_swing,data_motif \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
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
  stage-b-collapse-sweep)
    run_stage_b_collapse_sweep
    ;;
  stage-b-2file-brad-probe)
    run_stage_b_2file_brad_probe
    ;;
  stage-b-coverage-aware-probe)
    run_stage_b_coverage_aware_probe
    ;;
  stage-b-coverage-ab-sweep)
    run_stage_b_coverage_ab_sweep
    ;;
  stage-b-candidate-ranking)
    run_stage_b_candidate_ranking
    ;;
  stage-b-chord-aware-probe)
    run_stage_b_chord_aware_probe
    ;;
  stage-b-longer-phrase-probe)
    run_stage_b_longer_phrase_probe
    ;;
  stage-b-pitch-mode-compare)
    run_stage_b_pitch_mode_compare
    ;;
  stage-b-8bar-approach-phrase)
    run_stage_b_8bar_approach_phrase
    ;;
  stage-b-swing-motif-phrase)
    run_stage_b_swing_motif_phrase
    ;;
  stage-b-reference-stats)
    run_stage_b_reference_stats
    ;;
  stage-b-motif-templates)
    run_stage_b_motif_templates
    ;;
  stage-b-data-motif-compare)
    run_stage_b_data_motif_compare
    ;;
  stage-b-data-motif-review-export)
    run_stage_b_data_motif_review_export
    ;;
  stage-b-review-context-grid)
    run_stage_b_review_context_grid
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
