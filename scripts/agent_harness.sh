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
                Run a Stage B raw decode/generation gate probe.
  stage-b-raw-generation-repeatability
                Run raw Stage B generation across multiple seeds and source files.
  stage-b-dead-air-diagnostics
                Diagnose dead-air outliers from a Stage B generation probe report.
  stage-b-seed-strict-margin-diagnostics
                Diagnose per-seed strict margin risk from a Stage B repeatability summary.
  stage-b-margin-recovered-review-export
                Export review metrics from a margin-recovered Stage B repeatability summary.
  stage-b-margin-recovered-listening-notes
                Build pending listening notes for margin-recovered Stage B candidates.
  stage-b-margin-recovered-proxy-review-fill
                Fill margin-recovered listening notes with MIDI-metric proxy review.
  stage-b-margin-recovered-proxy-keep-focused-package
                Build a focused solo/context package for the margin-recovered proxy keep.
  stage-b-margin-recovered-focused-context-decision
                Review the margin-recovered focused package against solo/context MIDI metrics.
  stage-b-margin-recovered-focused-fallback-comparison
                Compare all margin-recovered candidates with focused context decisions.
  stage-b-margin-recovered-pitch-dead-air-repair
                Run expanded top-k sampling and select a pitch/dead-air repair candidate.
  stage-b-margin-recovered-pitch-vocab-sweep
                Run seed/top-k sweep and select a pitch-vocabulary qualified candidate.
  stage-b-margin-recovered-pitch-vocab-focused-context
                Package and review the selected pitch-vocabulary sweep candidate in context.
  stage-b-margin-recovered-pitch-vocab-focused-listening-notes
                Build focused listening notes for the selected pitch-vocabulary candidate.
  stage-b-margin-recovered-pitch-vocab-focused-listening-fill
                Fill the selected pitch-vocabulary focused listening review notes.
  stage-b-margin-recovered-timing-repetition-repair
                Run top-k/temperature repair sweep and select a timing/repetition improved candidate.
  stage-b-margin-recovered-timing-repetition-focused-context
                Package and review the selected timing/repetition repair candidate in context.
  stage-b-margin-recovered-timing-repetition-focused-listening-notes
                Build focused listening notes for the selected timing/repetition candidate.
  stage-b-margin-recovered-timing-repetition-focused-listening-fill
                Fill the selected timing/repetition focused listening review notes.
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
  stage-b-reference-pitch-roles
                Build reference pitch-role landing stats and compare latest hybrid candidates.
  stage-b-motif-templates
                Extract data-derived Stage B motif/rhythm templates.
  stage-b-data-motif-compare
                Compare hand-written swing against data-derived motif baseline.
  stage-b-data-motif-review-export
                Export named MIDI review candidates for data-derived motif compare.
  stage-b-review-context-grid
                Export review MIDI with chord context and straight-grid reference.
  stage-b-guide-tone-cadence
                Export straight-grid guide-tone/cadence candidates with chord context.
  stage-b-data-guide-hybrid
                Export data-motif rhythm plus guide-tone/cadence pitch candidates.
  stage-b-overlap-free-review-export
                Export overlap-free solo-line review MIDI variants and objective priority.
  stage-b-duration-variation-review
                Export varied-duration review candidates and objective priority.
  stage-b-phrase-cadence-review
                Export phrase/cadence review candidates and objective priority.
  stage-b-phrase-recovery-review
                Compare phrase/cadence against leap-recovery phrase candidates.
  stage-b-data-motif-phrase-recovery-review
                Compare data-motif guide tones against data-motif phrase recovery.
  stage-b-contour-landing-repair
                Compare data-motif phrase recovery against contour/cadence landing repair.
  stage-b-rhythm-phrase-variation
                Compare contour landing repair against rhythm/phrase vocabulary variation.
  stage-b-clean-review-package
                Extract objective-clean data-motif phrase recovery candidates for listening review.
  stage-b-proxy-keep-focused-package
                Extract proxy-keep reviewed candidates into a focused context listening package.
  stage-b-clean-context-diagnostics
                Diagnose objective-clean context MIDI candidates before subjective review.
  stage-b-clean-listening-review-notes
                Build clean listening review notes template for 3 context candidates.
  stage-b-focused-listening-review-notes
                Build focused listening review notes template for a proxy-keep package.
  manifest-dry-run
                Run audit -> manifest -> prepare_role_dataset smoke.
  chord-coverage-audit
                Audit chord annotation coverage in role metadata, sidecars, and MIDI text events.
  stage-b-chord-labeled-eval
                Evaluate the tiny chord-labeled subset manifest and pitch-role summary contract.
  stage-b-generated-chord-eval
                Evaluate generated candidate report bridge against known chord metadata.
  stage-b-data-guide-generated-chord-eval
                Generate data-guide hybrid review package and evaluate its known chord metadata.
  stage-b-review-markdown-chord-eval
                Generate data-guide hybrid review package and write a combined chord-eval review markdown.
  stage-b-listening-review-notes
                Generate pending listening review notes from the combined chord-eval review flow.
  stage-b-full-review-notes
                Generate pending listening review notes from the full data-guide review manifest.
  stage-b-objective-midi-review
                Run objective note-level MIDI review on the full data-guide review manifest.
  stage-b-objective-flags-review-flow
                Attach objective MIDI flags to review notes and aggregate review priority.
  stage-b-listening-review-aggregate
                Aggregate pending or filled listening review notes into next-step signals.
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
    scripts/run_stage_b_raw_generation_repeatability_sweep.py \
    scripts/diagnose_stage_b_dead_air_outliers.py \
    scripts/diagnose_stage_b_seed_strict_margins.py \
    scripts/build_stage_b_margin_recovered_review_export.py \
    scripts/build_stage_b_margin_recovered_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_proxy_review.py \
    scripts/build_stage_b_margin_recovered_focused_package.py \
    scripts/review_stage_b_margin_recovered_focused_context.py \
    scripts/select_stage_b_margin_recovered_repair_candidate.py \
    scripts/summarize_stage_b_margin_recovered_pitch_vocab_sweep.py \
    scripts/build_stage_b_margin_recovered_pitch_vocab_focused_package.py \
    scripts/build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    scripts/summarize_stage_b_margin_recovered_timing_repetition_repair.py \
    scripts/build_stage_b_margin_recovered_timing_repetition_focused_package.py \
    scripts/build_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
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
    scripts/audit_chord_progression_coverage.py \
    scripts/evaluate_chord_labeled_subset.py \
    scripts/evaluate_generated_candidate_chords.py \
    scripts/build_listening_review_notes.py \
    scripts/build_focused_listening_review_notes.py \
    scripts/summarize_listening_review_notes.py \
    scripts/build_focused_review_package.py \
    scripts/review_midi_note_objectives.py \
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
    --epochs 50 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 5 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8 \
    --top_k 4 \
    --postprocess_overlap \
    --require_note_groups \
    --require_valid_sample \
    --require_strict_valid_sample
}

run_stage_b_raw_generation_repeatability() {
  local run_id="${RUN_ID:-harness_stage_b_raw_generation_repeatability}"
  local issue_number="${ISSUE_NUMBER:-0}"
  local max_files="${MAX_FILES:-2}"
  local min_source_files="${MIN_SOURCE_FILES:-2}"
  local seeds="${SEEDS:-17,23,31}"
  local epochs="${EPOCHS:-50}"
  local batch_size="${BATCH_SIZE:-8}"
  local max_sequence="${MAX_SEQUENCE:-96}"
  local num_samples="${NUM_SAMPLES:-3}"
  local top_k="${TOP_K:-4}"
  local temperature="${TEMPERATURE:-0.9}"
  local max_dead_air_outlier_rate="${MAX_DEAD_AIR_OUTLIER_RATE:-0.25}"
  local warning_min_strict_samples_per_seed="${WARNING_MIN_STRICT_SAMPLES_PER_SEED:-2}"
  print_header "Stage B raw generation repeatability sweep"
  "$PYTHON_BIN" scripts/run_stage_b_raw_generation_repeatability_sweep.py \
    --run_id "$run_id" \
    --issue_number "$issue_number" \
    --max_files "$max_files" \
    --seeds "$seeds" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --max_sequence "$max_sequence" \
    --num_samples "$num_samples" \
    --top_k "$top_k" \
    --temperature "$temperature" \
    --min_seed_count 3 \
    --min_source_files "$min_source_files" \
    --min_strict_samples_per_seed 1 \
    --min_overall_strict_rate 0.67 \
    --warning_min_strict_samples_per_seed "$warning_min_strict_samples_per_seed" \
    --dead_air_gate 0.8 \
    --max_dead_air_outlier_rate "$max_dead_air_outlier_rate" \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_dead_air_diagnostics() {
  local run_id="${RUN_ID:-harness_stage_b_dead_air_diagnostics}"
  local report_path="${REPORT_PATH:-outputs/stage_b_generation_probe/issue_224_stage_b_raw_generation_repeatability_final2_seed31_files2/report.json}"
  print_header "Stage B dead-air diagnostics"
  "$PYTHON_BIN" scripts/diagnose_stage_b_dead_air_outliers.py \
    --run_id "$run_id" \
    --report_path "$report_path" \
    --expected_outliers 1
}

run_stage_b_seed_strict_margin_diagnostics() {
  local run_id="${RUN_ID:-harness_stage_b_seed_strict_margin_diagnostics}"
  local summary_path="${SUMMARY_PATH:-outputs/stage_b_raw_generation_repeatability/issue_232_stage_b_larger_source_risk_boundary_files6/repeatability_summary.json}"
  print_header "Stage B seed strict margin diagnostics"
  "$PYTHON_BIN" scripts/diagnose_stage_b_seed_strict_margins.py \
    --run_id "$run_id" \
    --summary_path "$summary_path" \
    --warning_min_strict_samples_per_seed 2 \
    --expected_margin_warning_seeds 17
}

run_stage_b_margin_recovered_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_review_export}"
  local summary_path="${SUMMARY_PATH:-outputs/stage_b_raw_generation_repeatability/issue_238_stage_b_candidate_count_margin_recovery/repeatability_summary.json}"
  print_header "Stage B margin-recovered candidate review export"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_review_export.py \
    --run_id "$run_id" \
    --summary_path "$summary_path" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_listening_notes() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_listening_notes}"
  local review_export="${REVIEW_EXPORT:-outputs/stage_b_margin_recovered_review_export/harness_stage_b_margin_recovered_review_export/candidate_review_export.json}"
  print_header "Stage B margin-recovered listening review notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_listening_notes.py \
    --run_id "$run_id" \
    --review_export "$review_export" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_proxy_review_fill() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_proxy_review}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_template.json}"
  print_header "Stage B margin-recovered MIDI proxy review fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_proxy_review.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_keep_candidate_id margin_recovered_rank_2_seed_31_sample_5
}

run_stage_b_margin_recovered_proxy_keep_focused_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_margin_recovered_proxy_review}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_proxy_keep_focused_package}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_proxy_review/${source_run_id}/listening_review_notes_proxy_filled.json}"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered MIDI proxy review fill"
    RUN_ID="$source_run_id" run_stage_b_margin_recovered_proxy_review_fill
  fi
  print_header "Stage B margin-recovered proxy keep focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_focused_package.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --min_candidates 1
}

run_stage_b_margin_recovered_focused_context_decision() {
  local focused_run_id="${FOCUSED_RUN_ID:-harness_stage_b_margin_recovered_proxy_keep_focused_package_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_focused_context_decision}"
  local focused_package="${FOCUSED_PACKAGE:-outputs/stage_b_margin_recovered_focused_package/${focused_run_id}/focused_review_package.json}"
  if [[ ! -f "$focused_package" ]]; then
    RUN_ID="$focused_run_id" run_stage_b_margin_recovered_proxy_keep_focused_package
  fi
  print_header "Stage B margin-recovered focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$run_id" \
    --focused_package "$focused_package" \
    --expected_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_focused_fallback_comparison() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_focused_fallback_package}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_focused_fallback_decision}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.json}"
  print_header "Stage B margin-recovered all-candidate focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_focused_package.py \
    --run_id "$package_run_id" \
    --review_notes "$review_notes" \
    --decision all \
    --min_candidates 3
  print_header "Stage B margin-recovered focused fallback comparison"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$run_id" \
    --focused_package "outputs/stage_b_margin_recovered_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_pitch_dead_air_repair() {
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_margin_recovered_pitch_dead_air_repair_generation}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_dead_air_repair_selection}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered pitch/dead-air repair generation"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$generation_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 31 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 12 \
    --temperature 0.9 \
    --top_k 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 254

  print_header "Stage B margin-recovered pitch/dead-air repair selection"
  "$PYTHON_BIN" scripts/select_stage_b_margin_recovered_repair_candidate.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${generation_run_id}/report.json" \
    --baseline_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --baseline_dead_air 0.4444444444444444 \
    --baseline_unique_pitch_count 4 \
    --expected_sample_index 8 \
    --require_partial_repair
}

run_stage_b_margin_recovered_pitch_vocab_sweep() {
  local seed17_run_id="${SEED17_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_seed17_topk5_temp090_n24}"
  local seed31_run_id="${SEED31_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_seed31_topk5_temp090_n24}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_sweep}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered pitch vocabulary sweep seed 17"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed17_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 17 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 24 \
    --temperature 0.9 \
    --top_k 5 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 256

  print_header "Stage B margin-recovered pitch vocabulary sweep seed 31"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed31_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 31 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 24 \
    --temperature 0.9 \
    --top_k 5 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 256

  print_header "Stage B margin-recovered pitch vocabulary sweep summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_pitch_vocab_sweep.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed17_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed31_run_id}/report.json" \
    --require_qualified \
    --expected_source_run_id "$seed17_run_id" \
    --expected_sample_index 4
}

run_stage_b_margin_recovered_pitch_vocab_focused_context() {
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_sweep}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_context_decision}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local sweep_summary="outputs/stage_b_margin_recovered_pitch_vocab_sweep/${sweep_run_id}/pitch_vocab_sweep_summary.json"
  if [[ ! -f "$sweep_summary" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary sweep"
    RUN_ID="$sweep_run_id" run_stage_b_margin_recovered_pitch_vocab_sweep
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_pitch_vocab_focused_package.py \
    --run_id "$package_run_id" \
    --sweep_summary "$sweep_summary" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered pitch vocabulary focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_pitch_vocab_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_notes}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local package_path="outputs/stage_b_margin_recovered_pitch_vocab_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_pitch_vocab_focused_context
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_pitch_vocab_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_fill}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local review_notes="outputs/stage_b_margin_recovered_pitch_vocab_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_timing_repetition_repair() {
  local seed37_run_id="${SEED37_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_seed37_topk7_temp086_n48}"
  local seed41_run_id="${SEED41_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_seed41_topk7_temp086_n48}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered timing/repetition repair seed 37"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed37_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 37 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.86 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 264

  print_header "Stage B margin-recovered timing/repetition repair seed 41"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed41_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 41 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.86 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 264

  print_header "Stage B margin-recovered timing/repetition repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_timing_repetition_repair.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed37_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed41_run_id}/report.json" \
    --require_qualified \
    --require_timing_repetition_improvement \
    --expected_source_run_id "$seed37_run_id" \
    --expected_sample_index 39
}

run_stage_b_margin_recovered_timing_repetition_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_repair}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_context_decision}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local repair_summary="outputs/stage_b_margin_recovered_timing_repetition_repair/${repair_run_id}/timing_repetition_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered timing/repetition repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_timing_repetition_repair
  fi
  print_header "Stage B margin-recovered timing/repetition focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_timing_repetition_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered timing/repetition focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_timing_repetition_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_timing_repetition_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local package_path="outputs/stage_b_margin_recovered_timing_repetition_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered timing/repetition focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_timing_repetition_focused_context
  fi
  print_header "Stage B margin-recovered timing/repetition focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_timing_repetition_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_fill}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local review_notes="outputs/stage_b_margin_recovered_timing_repetition_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered timing/repetition focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_timing_repetition_focused_listening_notes
  fi
  print_header "Stage B margin-recovered timing/repetition focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision needs_followup
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

run_stage_b_reference_pitch_roles() {
  local run_id="${RUN_ID:-harness_stage_b_reference_pitch_roles}"
  local generated_run_id="${run_id}_generated"
  print_header "Stage B generated hybrid candidate report"
  RUN_ID="$generated_run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B reference pitch-role landing statistics"
  "$PYTHON_BIN" scripts/run_stage_b_reference_stats.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --max_records 64 \
    --generated_report "outputs/stage_b_data_motif_compare/${generated_run_id}/data_motif_compare_report.json"
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

run_stage_b_guide_tone_cadence() {
  local run_id="${RUN_ID:-harness_stage_b_guide_tone_cadence}"
  print_header "Stage B straight-grid guide-tone/cadence review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif \
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

run_stage_b_data_guide_hybrid() {
  local run_id="${RUN_ID:-harness_stage_b_data_guide_hybrid}"
  print_header "Stage B data-motif rhythm plus guide-tone pitch review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
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

run_stage_b_overlap_free_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_overlap_free_review_export}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B overlap-free solo-line review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 97 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_duration_variation_review() {
  local run_id="${RUN_ID:-harness_stage_b_duration_variation_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B duration-variation review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 99 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes varied_grid,varied_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_phrase_cadence_review() {
  local run_id="${RUN_ID:-harness_stage_b_phrase_cadence_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B phrase/cadence review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 101 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes phrase_cadence,varied_guide_tones,data_motif,data_motif_guide_tones \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_phrase_recovery_review() {
  local run_id="${RUN_ID:-harness_stage_b_phrase_recovery_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B phrase recovery review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 105 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes phrase_cadence,phrase_recovery \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_data_motif_phrase_recovery_review() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_phrase_recovery_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B data-motif phrase recovery review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 107 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_guide_tones,data_motif_phrase_recovery,phrase_recovery \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_contour_landing_repair() {
  local run_id="${RUN_ID:-harness_stage_b_contour_landing_repair}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B contour/cadence landing repair review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 115 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_phrase_recovery,data_motif_contour_landing_repair \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_rhythm_phrase_variation() {
  local run_id="${RUN_ID:-harness_stage_b_rhythm_phrase_variation}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B rhythm/phrase vocabulary variation review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 118 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_contour_landing_repair,data_motif_rhythm_phrase_variation \
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
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_clean_review_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_data_motif_phrase_recovery_review}"
  local run_id="${RUN_ID:-harness_stage_b_clean_review_package}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${source_run_id}/review_manifest.json"
  local objective_report_path="outputs/stage_b_objective_midi_review/${source_run_id}/objective_midi_note_review.json"
  if [[ ! -f "$review_manifest_path" || ! -f "$objective_report_path" ]]; then
    print_header "Stage B data-motif phrase recovery review export"
    SOURCE_RUN_ID="$source_run_id" RUN_ID="$source_run_id" run_stage_b_data_motif_phrase_recovery_review
  fi
  print_header "Stage B objective-clean review package"
  "$PYTHON_BIN" scripts/build_clean_review_package.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --objective_report "$objective_report_path" \
    --allowed_modes data_motif_phrase_recovery \
    --copy_files \
    --min_candidates 3
}

run_stage_b_proxy_keep_focused_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_phrase_shape_tension_proxy}"
  local objective_run_id="${OBJECTIVE_RUN_ID:-harness_stage_b_rhythm_phrase_variation}"
  local run_id="${RUN_ID:-harness_stage_b_proxy_keep_focused_package}"
  local review_notes_file="${REVIEW_NOTES_FILE:-phrase_shape_tension_repaired_review_notes_midi_proxy.json}"
  local review_notes_path="${REVIEW_NOTES_PATH:-outputs/stage_b_listening_review_notes/${source_run_id}/${review_notes_file}}"
  local objective_report_path="outputs/stage_b_objective_midi_review/${objective_run_id}/objective_midi_note_review.json"
  if [[ ! -f "$review_notes_path" ]]; then
    printf 'Missing proxy-filled review notes: %s\n' "$review_notes_path" >&2
    return 2
  fi
  if [[ ! -f "$objective_report_path" ]]; then
    printf 'Missing objective MIDI review report: %s\n' "$objective_report_path" >&2
    return 2
  fi
  print_header "Stage B proxy-keep focused review package"
  "$PYTHON_BIN" scripts/build_focused_review_package.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path" \
    --objective_report "$objective_report_path" \
    --copy_files \
    --min_candidates 1
}

run_stage_b_clean_context_diagnostics() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_clean_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_clean_context_diagnostics}"
  local clean_package_path="outputs/stage_b_clean_review_package/${source_run_id}/clean_review_package.json"
  if [[ ! -f "$clean_package_path" ]]; then
    print_header "Stage B objective-clean review package"
    RUN_ID="$source_run_id" run_stage_b_clean_review_package
  fi
  print_header "Stage B clean context diagnostics"
  "$PYTHON_BIN" scripts/build_clean_context_diagnostics.py \
    --run_id "$run_id" \
    --clean_package "$clean_package_path" \
    --min_candidates 3
}


run_stage_b_clean_listening_review_notes() {
  local clean_run_id="${CLEAN_RUN_ID:-harness_stage_b_clean_review_package}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_clean_context_diagnostics}"
  local run_id="${RUN_ID:-harness_stage_b_clean_listening_review_notes}"
  local clean_package_path="outputs/stage_b_clean_review_package/${clean_run_id}/clean_review_package.json"
  local diagnostics_path="outputs/stage_b_clean_context_diagnostics/${diagnostics_run_id}/clean_context_diagnostics.json"
  if [[ ! -f "$clean_package_path" ]]; then
    RUN_ID="$clean_run_id" run_stage_b_clean_review_package
  fi
  if [[ ! -f "$diagnostics_path" ]]; then
    SOURCE_RUN_ID="$clean_run_id" RUN_ID="$diagnostics_run_id" run_stage_b_clean_context_diagnostics
  fi
  print_header "Stage B clean listening review notes"
  "$PYTHON_BIN" scripts/build_clean_listening_review_notes.py     --run_id "$run_id"     --clean_package "$clean_package_path"     --clean_context_diagnostics "$diagnostics_path"
}

run_stage_b_focused_listening_review_notes() {
  local focused_run_id="${FOCUSED_RUN_ID:-harness_stage_b_register_safe_proxy_keep_focused_package}"
  local run_id="${RUN_ID:-harness_stage_b_focused_listening_review_notes}"
  local focused_package_path="${FOCUSED_PACKAGE_PATH:-outputs/stage_b_focused_review_package/${focused_run_id}/focused_review_package.json}"
  if [[ ! -f "$focused_package_path" ]]; then
    printf 'Missing focused review package: %s\n' "$focused_package_path" >&2
    return 2
  fi
  print_header "Stage B focused listening review notes"
  "$PYTHON_BIN" scripts/build_focused_listening_review_notes.py \
    --run_id "$run_id" \
    --focused_package "$focused_package_path"
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

run_chord_coverage_audit() {
  local run_id="${RUN_ID:-harness_chord_coverage_audit}"
  print_header "Chord progression coverage audit"
  "$PYTHON_BIN" scripts/audit_chord_progression_coverage.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset \
    --role_meta_roots ./data,./outputs \
    --max_midi_files 120
}

run_stage_b_chord_labeled_eval() {
  local run_id="${RUN_ID:-harness_stage_b_chord_labeled_eval}"
  print_header "Stage B chord-labeled evaluation subset"
  "$PYTHON_BIN" scripts/evaluate_chord_labeled_subset.py \
    --run_id "$run_id" \
    --manifest data/eval/stage_b_chord_labeled_tiny/manifest.json \
    --min_samples 2 \
    --min_notes 16
}

run_stage_b_generated_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_generated_chord_eval}"
  print_header "Stage B generated candidate chord-labeled eval bridge"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --write_tiny_fixture \
    --candidate_limit 1
}

run_stage_b_data_guide_generated_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_data_guide_generated_chord_eval}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B data-guide hybrid generated chord eval"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --candidate_report "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --candidate_limit 6
}

run_stage_b_review_markdown_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_review_markdown_chord_eval}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B combined review markdown with chord eval"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --candidate_report "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --review_markdown "outputs/stage_b_data_motif_review/${run_id}/review_candidates.md" \
    --candidate_limit 6
}

run_stage_b_listening_review_notes() {
  local run_id="${RUN_ID:-harness_stage_b_listening_review_notes}"
  print_header "Stage B combined review markdown with chord eval"
  RUN_ID="$run_id" run_stage_b_review_markdown_chord_eval
  print_header "Stage B listening review notes template"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --generated_chord_eval_report "outputs/stage_b_generated_chord_eval/${run_id}/generated_chord_eval_report.json" \
    --source_review_markdown "outputs/stage_b_generated_chord_eval/${run_id}/review_candidates_with_chord_eval.md"
}

run_stage_b_full_review_notes() {
  local run_id="${RUN_ID:-harness_stage_b_full_review_notes}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B full review manifest listening notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --source_review_markdown "outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
}

run_stage_b_objective_midi_review() {
  local run_id="${RUN_ID:-harness_stage_b_objective_midi_review}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
}

run_stage_b_objective_flags_review_flow() {
  local run_id="${RUN_ID:-harness_stage_b_objective_flags_review_flow}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_listening_review_aggregate() {
  local run_id="${RUN_ID:-harness_stage_b_listening_review_aggregate}"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  if [[ ! -f "$review_notes_path" || "${FORCE_REGEN:-0}" == "1" ]]; then
    print_header "Stage B listening review notes template"
    RUN_ID="$run_id" run_stage_b_listening_review_notes
  else
    print_header "Stage B listening review notes template"
    printf 'Using existing review notes: %s\n' "$review_notes_path"
  fi
  print_header "Stage B listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
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
  stage-b-raw-generation-repeatability)
    run_stage_b_raw_generation_repeatability
    ;;
  stage-b-dead-air-diagnostics)
    run_stage_b_dead_air_diagnostics
    ;;
  stage-b-seed-strict-margin-diagnostics)
    run_stage_b_seed_strict_margin_diagnostics
    ;;
  stage-b-margin-recovered-review-export)
    run_stage_b_margin_recovered_review_export
    ;;
  stage-b-margin-recovered-listening-notes)
    run_stage_b_margin_recovered_listening_notes
    ;;
  stage-b-margin-recovered-proxy-review-fill)
    run_stage_b_margin_recovered_proxy_review_fill
    ;;
  stage-b-margin-recovered-proxy-keep-focused-package)
    run_stage_b_margin_recovered_proxy_keep_focused_package
    ;;
  stage-b-margin-recovered-focused-context-decision)
    run_stage_b_margin_recovered_focused_context_decision
    ;;
  stage-b-margin-recovered-focused-fallback-comparison)
    run_stage_b_margin_recovered_focused_fallback_comparison
    ;;
  stage-b-margin-recovered-pitch-dead-air-repair)
    run_stage_b_margin_recovered_pitch_dead_air_repair
    ;;
  stage-b-margin-recovered-pitch-vocab-sweep)
    run_stage_b_margin_recovered_pitch_vocab_sweep
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-context)
    run_stage_b_margin_recovered_pitch_vocab_focused_context
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-listening-notes)
    run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-listening-fill)
    run_stage_b_margin_recovered_pitch_vocab_focused_listening_fill
    ;;
  stage-b-margin-recovered-timing-repetition-repair)
    run_stage_b_margin_recovered_timing_repetition_repair
    ;;
  stage-b-margin-recovered-timing-repetition-focused-context)
    run_stage_b_margin_recovered_timing_repetition_focused_context
    ;;
  stage-b-margin-recovered-timing-repetition-focused-listening-notes)
    run_stage_b_margin_recovered_timing_repetition_focused_listening_notes
    ;;
  stage-b-margin-recovered-timing-repetition-focused-listening-fill)
    run_stage_b_margin_recovered_timing_repetition_focused_listening_fill
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
  stage-b-reference-pitch-roles)
    run_stage_b_reference_pitch_roles
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
  stage-b-guide-tone-cadence)
    run_stage_b_guide_tone_cadence
    ;;
  stage-b-data-guide-hybrid)
    run_stage_b_data_guide_hybrid
    ;;
  stage-b-overlap-free-review-export)
    run_stage_b_overlap_free_review_export
    ;;
  stage-b-duration-variation-review)
    run_stage_b_duration_variation_review
    ;;
  stage-b-phrase-cadence-review)
    run_stage_b_phrase_cadence_review
    ;;
  stage-b-phrase-recovery-review)
    run_stage_b_phrase_recovery_review
    ;;
  stage-b-data-motif-phrase-recovery-review)
    run_stage_b_data_motif_phrase_recovery_review
    ;;
  stage-b-contour-landing-repair)
    run_stage_b_contour_landing_repair
    ;;
  stage-b-rhythm-phrase-variation)
    run_stage_b_rhythm_phrase_variation
    ;;
  stage-b-clean-review-package)
    run_stage_b_clean_review_package
    ;;
  stage-b-proxy-keep-focused-package)
    run_stage_b_proxy_keep_focused_package
    ;;
  stage-b-clean-context-diagnostics)
    run_stage_b_clean_context_diagnostics
    ;;
  stage-b-clean-listening-review-notes)
    run_stage_b_clean_listening_review_notes
    ;;
  stage-b-focused-listening-review-notes)
    run_stage_b_focused_listening_review_notes
    ;;
  manifest-dry-run)
    run_manifest_dry_run
    ;;
  chord-coverage-audit)
    run_chord_coverage_audit
    ;;
  stage-b-chord-labeled-eval)
    run_stage_b_chord_labeled_eval
    ;;
  stage-b-generated-chord-eval)
    run_stage_b_generated_chord_eval
    ;;
  stage-b-data-guide-generated-chord-eval)
    run_stage_b_data_guide_generated_chord_eval
    ;;
  stage-b-review-markdown-chord-eval)
    run_stage_b_review_markdown_chord_eval
    ;;
  stage-b-listening-review-notes)
    run_stage_b_listening_review_notes
    ;;
  stage-b-full-review-notes)
    run_stage_b_full_review_notes
    ;;
  stage-b-objective-midi-review)
    run_stage_b_objective_midi_review
    ;;
  stage-b-objective-flags-review-flow)
    run_stage_b_objective_flags_review_flow
    ;;
  stage-b-listening-review-aggregate)
    run_stage_b_listening_review_aggregate
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
