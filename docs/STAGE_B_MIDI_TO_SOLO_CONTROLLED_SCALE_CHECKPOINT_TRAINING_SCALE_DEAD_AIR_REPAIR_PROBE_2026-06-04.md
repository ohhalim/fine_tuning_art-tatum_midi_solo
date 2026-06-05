# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe`
- selected-scale dead-air target qualified: `true`
- MIDI-to-solo musical quality claimed: `false`

## Source Repeatability

- seed count: `3`
- sample count: `9`
- valid / strict / grammar: `2` / `2` / `9`
- note-count / grammar / dead-air / collapse failure count: `0` / `0` / `7` / `0`
- avg onset / sustained coverage ratio: `0.4548611111111111` / `0.625`

## Repair Result

- constrained note groups per bar: `12`
- sample count: `3`
- valid / strict / grammar: `3` / `3` / `3`
- note-count / grammar / dead-air / collapse failure count: `0` / `0` / `0` / `0`
- avg onset / sustained coverage ratio: `0.5729166666666666` / `0.7083333333333334`
- avg / max postprocess removal ratio: `0.3888888888888889` / `0.4166666666666667`

## Delta

- dead-air failure delta: `7`
- valid sample rate delta: `0.7777777777777778`
- strict sample rate delta: `0.7777777777777778`
- onset / sustained coverage delta: `0.11805555555555552` / `0.08333333333333337`
- postprocess removal delta: `0.19444444444444445`

## Failure Reasons

- none

## Not Proven

- `repeatability`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
