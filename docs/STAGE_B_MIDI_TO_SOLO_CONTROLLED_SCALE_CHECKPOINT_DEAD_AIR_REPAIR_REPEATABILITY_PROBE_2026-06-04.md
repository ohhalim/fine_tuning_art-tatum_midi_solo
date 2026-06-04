# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Repair Repeatability Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision`
- seed count: `3`
- sample count: `9`
- dead-air repair repeatability target qualified: `false`
- MIDI-to-solo musical quality claimed: `false`

## Aggregate

- valid / strict / grammar: `7` / `7` / `9`
- all seed gate passed: `true`
- all samples strict valid: `false`
- collapse warning sample count: `1`
- avg postprocess removal ratio: `0.375`
- avg onset / sustained coverage ratio: `0.5486111111111112` / `0.7222222222222222`

## Delta

- source / repeatability sample count: `3` / `9`
- strict valid sample delta: `4`
- postprocess removal delta: `0.041666666666666685`

## Seed Rows

- seed `44`: valid/strict/grammar `3`/`3`/`3`, collapse `0`
- seed `52`: valid/strict/grammar `3`/`3`/`3`, collapse `0`
- seed `60`: valid/strict/grammar `1`/`1`/`3`, collapse `1`

## Failure Reasons

- `dead-air ratio too high: 0.800 >= 0.800; collapse=postprocess_removed_majority`: `1`
- `dead-air ratio too high: 0.846 >= 0.800`: `1`

## Not Proven

- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
