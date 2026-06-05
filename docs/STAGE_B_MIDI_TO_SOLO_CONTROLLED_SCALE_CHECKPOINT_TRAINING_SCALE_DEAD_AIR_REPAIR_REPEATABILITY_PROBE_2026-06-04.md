# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repair Repeatability Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision`
- seed count: `3`
- sample count: `9`
- selected-scale dead-air repair repeatability target qualified: `false`
- MIDI-to-solo musical quality claimed: `false`

## Aggregate

- valid / strict / grammar: `7` / `7` / `9`
- note-count / grammar / dead-air / collapse failure count: `0` / `0` / `2` / `1`
- all seed gate passed: `true`
- all samples strict valid: `false`
- avg postprocess removal ratio: `0.412037037037037`
- avg onset / sustained coverage ratio: `0.5520833333333334` / `0.7222222222222222`

## Delta

- source / repeatability sample count: `3` / `9`
- strict valid sample delta: `4`
- dead-air failure delta: `2`
- collapse warning delta: `1`
- postprocess removal delta: `0.023148148148148084`

## Seed Rows

- seed `47`: valid/strict/grammar `3`/`3`/`3`, dead-air `0`, collapse `0`
- seed `52`: valid/strict/grammar `1`/`1`/`3`, dead-air `2`, collapse `1`
- seed `60`: valid/strict/grammar `3`/`3`/`3`, dead-air `0`, collapse `0`

## Failure Reasons

- `dead-air ratio too high: 0.833 >= 0.800`: `1`
- `dead-air ratio too high: 1.000 >= 0.800; collapse=postprocess_removed_majority`: `1`

## Not Proven

- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
