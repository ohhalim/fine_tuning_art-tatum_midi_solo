# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Decision

## Summary

- boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep`
- auto progress allowed: `true`
- musical quality claimed: `false`

## Observed Failure

- note count: `9`
- pitch range: `29-89`
- pitch span: `60`
- max abs interval: `60`
- large interval ratio: `0.875`
- severe interval count: `6`
- intervals: `[15, -24, 60, -60, 34, -3, 27, -34]`

## Guard Targets

- max pitch span: `24`
- max abs interval: `12`
- max large interval ratio: `0.35`
- max severe interval count: `0`
- preferred pitch range: `48-84`

## Repair Targets

- `filter_pitch_candidates_to_preferred_solo_range`
- `reject_or_repair_adjacent_interval_above_target`
- `penalize_large_register_jumps_during_candidate_ranking`
- `require_small_leap_or_stepwise_support_ratio`
- `fail_audio_package_when_range_interval_guard_fails`

## Not Proven

- `repaired_candidate_exists`
- `audio_rendered_quality`
- `musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
