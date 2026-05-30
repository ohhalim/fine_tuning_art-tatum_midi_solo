# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sweep

## Summary

- boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package`
- target passed: `true`
- target qualified count: `3`
- candidate count: `48`
- generation success: `true`
- all samples grammar valid: `true`
- musical quality claimed: `false`

## Generation Runs

| interval cap | samples | valid | strict | grammar | target qualified | collapse warning rate |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 5 | 4 | 12 | 0 | 0.417 |
| 9 | 12 | 6 | 5 | 12 | 2 | 0.583 |
| 7 | 12 | 3 | 2 | 12 | 1 | 0.667 |
| 5 | 12 | 2 | 1 | 12 | 0 | 0.833 |

## Top Candidates

| rank | cap | seed | sample | target | notes | coverage | tail | postprocess removal | pitch range | span | max interval | large ratio | failures | midi |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| 1 | 9 | 70 | 9 | true | 11 | 1.000 | 0 | 0.312 | 53-74 | 21 | 9 | 0.000 | none | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_9_seed_62/samples/stage_b_sample_9.mid |
| 2 | 7 | 66 | 5 | true | 9 | 0.906 | 0 | 0.438 | 51-63 | 12 | 12 | 0.125 | none | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_7_seed_62/samples/stage_b_sample_5.mid |
| 3 | 9 | 62 | 1 | true | 9 | 0.875 | 1 | 0.438 | 51-68 | 17 | 12 | 0.125 | none | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_9_seed_62/samples/stage_b_sample_1.mid |
| 4 | 7 | 62 | 1 | false | 7 | 0.875 | 0 | 0.562 | 58-67 | 9 | 5 | 0.000 | strict_valid_failed, note_count_below_target, postprocess_removal_above_target | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_7_seed_62/samples/stage_b_sample_1.mid |
| 5 | 5 | 65 | 4 | false | 7 | 0.812 | 0 | 0.562 | 58-67 | 9 | 7 | 0.000 | strict_valid_failed, note_count_below_target, phrase_coverage_below_target, postprocess_removal_above_target | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_5_seed_62/samples/stage_b_sample_4.mid |
| 6 | 5 | 69 | 8 | false | 8 | 0.781 | 0 | 0.500 | 50-63 | 13 | 8 | 0.000 | strict_valid_failed, phrase_coverage_below_target, postprocess_removal_above_target | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_5_seed_62/samples/stage_b_sample_8.mid |
| 7 | 7 | 63 | 2 | false | 6 | 0.781 | 0 | 0.625 | 48-68 | 20 | 8 | 0.000 | strict_valid_failed, note_count_below_target, phrase_coverage_below_target, postprocess_removal_above_target | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_7_seed_62/samples/stage_b_sample_2.mid |
| 8 | 5 | 66 | 5 | false | 10 | 1.000 | 0 | 0.375 | 48-60 | 12 | 9 | 0.000 | strict_valid_failed | outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/generation_probe/cap_5_seed_62/samples/stage_b_sample_5.mid |

## Not Proven

- `audio_rendered_quality`
- `human_audio_keep`
- `musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
