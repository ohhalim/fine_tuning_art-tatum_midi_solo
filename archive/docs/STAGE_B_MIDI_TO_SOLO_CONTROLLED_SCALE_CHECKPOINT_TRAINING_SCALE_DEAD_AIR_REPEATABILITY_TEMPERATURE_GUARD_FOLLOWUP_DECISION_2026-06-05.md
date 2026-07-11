# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repeatability Temperature Guard Follow-Up Decision

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe`
- selected target: `postprocess_removal_dead_air_repair`
- source / repair strict shortfall: `2` -> `1`
- source / repair dead-air failure: `2` -> `1`
- source / repair collapse warning: `1` -> `0`
- MIDI-to-solo musical quality claimed: `false`

## Evidence

- seed count: `3`
- sample count: `9`
- valid / strict / grammar: `8` / `8` / `9`
- note-count / grammar / dead-air / collapse failure count: `0` / `0` / `1` / `0`
- failed seeds: `[52]`
- avg postprocess removal ratio: `0.3611111111111111`
- max failed-seed avg postprocess removal ratio: `0.3888888888888889`
- avg onset / sustained coverage ratio: `0.5763888888888888` / `0.7222222222222222`
- temperature / top_k: `0.75` / `4`

## Failure Reasons

- `dead-air ratio too high: 0.846 >= 0.800`: `1`

## Decision

- temperature follow-up selected: `false`
- top_k follow-up selected: `false`
- postprocess removal repair selected: `true`
- coverage repair selected: `true`
- audio review selected: `false`
- additional training scale selected: `false`
- critical user input required: `false`

## Repair Config

- source_temperature: `0.75`
- top_k: `4`
- seeds: `[47, 52, 60]`
- num_samples: `3`
- max_sequence: `160`
- constrained_note_groups_per_bar: `12`
- coverage_position_window: `1`
- chord_pitch_mode: `approach_tensions`
- jazz_rhythm_profile: `swing_motif`
- max_simultaneous_notes: `1`
- target_avg_postprocess_removal_ratio: `0.3`
- target_dead_air_failure_count: `0`
- strategy: `reduce_overlap_before_postprocess_then_verify_dead_air`

## Not Proven

- `postprocess_removal_repair_result`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
