# Stage B Generic Tiny Checkpoint Repair Phrase Continuation MIDI Note Failure Review

## Summary

- boundary: `generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all`
- overall decision: `reject_all`
- candidate decision: `reject`
- primary failure: `midi_note_random_large_leaps`
- failed candidate count: `1/1`
- repair target: `range_interval_guard_missing`
- human/audio keep claimed: `false`
- musical quality claimed: `false`
- auto progress allowed: `true`

## MIDI Note Audit

| rank | seed | sample | notes | pitch range | span | max interval | large interval ratio | severe intervals | failure reasons |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 62 | 1 | 9 | 29-89 | 60 | 60 | 0.875 | 6 | pitch_span_above_target, max_interval_above_target, large_interval_ratio_above_target, severe_interval_present |

## Note Sequence

### Rank 1

| index | start | duration | pitch | name |
|---:|---:|---:|---:|---|
| 1 | 0.250 | 0.250 | 38 | D2 |
| 2 | 1.750 | 0.250 | 53 | F3 |
| 3 | 2.000 | 0.500 | 29 | F1 |
| 4 | 3.500 | 0.250 | 89 | F6 |
| 5 | 3.750 | 1.000 | 29 | F1 |
| 6 | 5.500 | 0.250 | 63 | D#4 |
| 7 | 5.750 | 0.500 | 60 | C4 |
| 8 | 6.250 | 0.750 | 87 | D#6 |
| 9 | 7.250 | 0.250 | 53 | F3 |

- intervals: `[15, -24, 60, -60, 34, -3, 27, -34]`

## Repair Targets

- `constrain_solo_pitch_range`
- `limit_max_adjacent_interval`
- `penalize_severe_register_jumps`
- `require_step_or_small_leap_contour_support`

## Not Proven

- `human_audio_keep`
- `audio_rendered_quality`
- `musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
