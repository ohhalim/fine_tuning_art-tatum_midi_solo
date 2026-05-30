# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Rejection Analysis

## Summary

- boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis`
- source boundary: `generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_review_reject_all`
- analyzed candidates: `3`
- common evidence flags: `high_dead_air_or_sparse_phrase`
- primary next repair target: `sparse_phrase_continuity_after_range_interval_guard`
- quality cause claim: `not_claimed`
- musical quality claimed: `false`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision`

## Evidence Flag Counts

| flag | count |
|---|---:|
| `adjacent_pitch_repeat_present` | 2 |
| `compressed_pitch_vocabulary` | 1 |
| `guard_edge_interval_present` | 1 |
| `high_dead_air_or_sparse_phrase` | 3 |
| `long_internal_gap_present` | 2 |
| `octave_or_larger_interval_present` | 2 |
| `pitch_cell_repetition_present` | 1 |
| `repetitive_duration_profile` | 1 |
| `two_note_oscillation_present` | 1 |

## Candidates

| rank | notes | unique | gap ratio | max gap | max interval | adjacent repeat | two-note windows | flags |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 11 | 8 | 0.4688 | 1.5000 | 9 | 1 | 1 | `high_dead_air_or_sparse_phrase`, `long_internal_gap_present`, `adjacent_pitch_repeat_present`, `two_note_oscillation_present`, `pitch_cell_repetition_present`, `guard_edge_interval_present`, `repetitive_duration_profile` |
| 2 | 9 | 6 | 0.4688 | 0.7500 | 12 | 1 | 0 | `high_dead_air_or_sparse_phrase`, `adjacent_pitch_repeat_present`, `octave_or_larger_interval_present`, `compressed_pitch_vocabulary` |
| 3 | 9 | 7 | 0.5312 | 1.2500 | 12 | 0 | 0 | `high_dead_air_or_sparse_phrase`, `long_internal_gap_present`, `octave_or_larger_interval_present` |

## Not Proven

- `musical_quality`
- `quality_root_cause`
- `multi_reviewer_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
