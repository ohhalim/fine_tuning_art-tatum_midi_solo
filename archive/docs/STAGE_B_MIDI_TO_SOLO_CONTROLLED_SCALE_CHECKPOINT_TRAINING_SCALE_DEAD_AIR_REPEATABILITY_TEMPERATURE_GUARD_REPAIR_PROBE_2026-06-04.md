# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repeatability Temperature Guard Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision`
- selected-scale temperature guard repair target qualified: `false`
- source / repair temperature: `0.9` / `0.75`
- top_k: `4`
- MIDI-to-solo musical quality claimed: `false`

## Aggregate

- seed count: `3`
- sample count: `9`
- valid / strict / grammar: `8` / `8` / `9`
- note-count / grammar / dead-air / collapse failure count: `0` / `0` / `1` / `0`
- all seed gate passed: `true`
- all samples strict valid: `false`
- avg postprocess removal ratio: `0.3611111111111111`
- avg onset / sustained coverage ratio: `0.5763888888888888` / `0.7222222222222222`

## Delta

- strict valid sample delta: `1`
- strict sample shortfall: `2` -> `1`
- dead-air failure count: `2` -> `1`
- collapse warning sample count: `1` -> `0`
- postprocess removal delta: `-0.050925925925925875`
- onset / sustained coverage delta: `0.02430555555555547` / `0.0`

## Seed Rows

- seed `47`: valid/strict/grammar `3`/`3`/`3`, dead-air `0`, collapse `0`
- seed `52`: valid/strict/grammar `2`/`2`/`3`, dead-air `1`, collapse `0`
- seed `60`: valid/strict/grammar `3`/`3`/`3`, dead-air `0`, collapse `0`

## Failure Reasons

- `dead-air ratio too high: 0.846 >= 0.800`: `1`

## Not Proven

- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
