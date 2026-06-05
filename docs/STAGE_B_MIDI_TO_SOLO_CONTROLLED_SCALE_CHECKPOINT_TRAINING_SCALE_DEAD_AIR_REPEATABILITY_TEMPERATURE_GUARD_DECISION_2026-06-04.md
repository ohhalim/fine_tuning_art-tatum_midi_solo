# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repeatability Temperature Guard Decision

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe`
- selected target: `lower_temperature_repeatability_guard_repair`
- source temperature / top_k: `0.9` / `4`
- selected temperature / top_k: `0.75` / `4`
- MIDI-to-solo musical quality claimed: `false`

## Evidence

- seed count: `3`
- sample count: `9`
- valid / strict / grammar: `7` / `7` / `9`
- strict sample shortfall: `2`
- failed seeds: `[52]`
- dead-air failure count: `2`
- collapse warning sample count: `1`
- avg postprocess removal ratio: `0.412037037037037`
- avg onset / sustained coverage ratio: `0.5520833333333334` / `0.7222222222222222`

## Failure Reasons

- `dead-air ratio too high: 0.833 >= 0.800`: `1`
- `dead-air ratio too high: 1.000 >= 0.800; collapse=postprocess_removed_majority`: `1`

## Decision

- temperature change selected: `true`
- top_k change selected: `false`
- audio review selected: `false`
- training scale change selected: `false`
- critical user input required: `false`

## Guard Config

- temperature: `0.75`
- top_k: `4`
- seeds: `[47, 52, 60]`
- num_samples: `3`
- max_sequence: `160`
- constrained_note_groups_per_bar: `12`
- coverage_position_window: `1`
- chord_pitch_mode: `approach_tensions`
- jazz_rhythm_profile: `swing_motif`
- max_simultaneous_notes: `1`

## Not Proven

- `guard_probe_result`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
