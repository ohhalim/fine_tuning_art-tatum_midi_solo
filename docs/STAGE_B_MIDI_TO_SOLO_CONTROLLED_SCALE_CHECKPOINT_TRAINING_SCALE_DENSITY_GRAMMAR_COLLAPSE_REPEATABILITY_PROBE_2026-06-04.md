# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Density Grammar Collapse Repeatability Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision`
- seed count: `3`
- sample count: `9`
- density/grammar/collapse repeatability target supported: `true`
- strict gate stable: `false`
- dead-air remaining: `true`
- MIDI-to-solo musical quality claimed: `false`

## Aggregate

- valid / strict / grammar: `2` / `2` / `9`
- note-count / grammar / dead-air failure count: `0` / `0` / `7`
- all seed gate passed: `false`
- all samples strict valid: `false`
- collapse warning sample count: `0`
- avg postprocess removal ratio: `0.19444444444444445`
- avg onset / sustained coverage ratio: `0.4548611111111111` / `0.625`

## Delta

- source / repeatability sample count: `3` / `9`
- strict valid sample delta: `1`
- postprocess removal delta: `0.0069444444444444475`

## Seed Rows

- seed `47`: valid/strict/grammar `1`/`1`/`3`, collapse `0`
- seed `52`: valid/strict/grammar `0`/`0`/`3`, collapse `0`
- seed `60`: valid/strict/grammar `1`/`1`/`3`, collapse `0`

## Failure Reasons

- `dead-air ratio too high: 0.833 >= 0.800`: `2`
- `dead-air ratio too high: 0.818 >= 0.800`: `1`
- `dead-air ratio too high: 0.846 >= 0.800`: `1`
- `dead-air ratio too high: 1.000 >= 0.800`: `1`
- `dead-air ratio too high: 0.909 >= 0.800`: `1`
- `dead-air ratio too high: 0.917 >= 0.800`: `1`

## Not Proven

- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
