# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Decision

## Summary

- input boundary: `generic_tiny_checkpoint_repair_audio_review_reject_all`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep`
- auto progress allowed: `true`
- critical user input required: `false`
- human/audio keep claimed: `false`
- musical quality claimed: `false`

## User Review

- overall decision: `reject_all`
- candidate decision: `reject`
- primary failure: `plunk_and_stop`
- timing: `too_short_or_stiff`
- phrase: `fragmented`
- vocabulary: `not_musical`
- assessment: all candidates only plunk briefly and end

## Repair Targets

- `increase_min_note_events_per_review_window`
- `require_phrase_continuation_after_initial_cell`
- `limit_terminal_dead_air_after_last_note`
- `penalize_single_cell_or_two_hit_outputs`
- `require_cadence_or_contour_resolution_before_end`
- `prefer_motif_extension_over_isolated_hits`

## Not Proven

- `human_audio_keep`
- `musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
