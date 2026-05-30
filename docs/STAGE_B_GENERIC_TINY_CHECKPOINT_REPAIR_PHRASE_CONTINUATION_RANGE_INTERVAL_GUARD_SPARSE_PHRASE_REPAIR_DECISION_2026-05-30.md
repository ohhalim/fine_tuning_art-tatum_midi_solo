# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Repair Decision

## Summary

- boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision`
- source boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis`
- decision: `run_sparse_phrase_repair_sweep`
- primary repair target: `sparse_phrase_continuity_after_range_interval_guard`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep`
- musical quality claimed: `false`
- quality cause claimed: `false`

## Observed Evidence

- candidate count: `3`
- gap ratio min / avg / max: `0.4688` / `0.4896` / `0.5312`
- max internal gap min / avg / max: `0.75` / `1.1667` / `1.5`
- adjacent repeat candidates: `2`
- octave-or-larger interval candidates: `2`

## Target Thresholds

| target | value |
|---|---:|
| `max_gap_ratio_to_window` | 0.4 |
| `max_internal_gap_beats` | 0.75 |
| `min_note_count` | 10 |
| `min_phrase_coverage_ratio` | 0.9 |
| `max_tail_empty_steps` | 0 |
| `max_abs_interval` | 12 |

## Planned Sweep Controls

| control | value |
|---|---|
| `interval_caps` | `[9, 7, 5]` |
| `coverage_aware_positions` | `True` |
| `coverage_position_window` | `0` |
| `keep_range_interval_guard` | `True` |
| `rank_by_gap_ratio_and_internal_gap` | `True` |
| `reject_adjacent_pitch_repeats_when_possible` | `True` |

## Not Proven

- `sparse_phrase_repair_candidate_exists`
- `human_audio_keep`
- `musical_quality`
- `quality_root_cause`
- `broad_trained_model_quality`
- `brad_style_adaptation`
