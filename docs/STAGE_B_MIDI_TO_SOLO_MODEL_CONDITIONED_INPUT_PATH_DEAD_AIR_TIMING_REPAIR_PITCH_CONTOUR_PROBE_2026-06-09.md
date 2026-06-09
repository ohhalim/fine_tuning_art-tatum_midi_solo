# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Pitch Contour Probe

## Summary

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package`
- repair passed: `true`
- source candidate count: `3`
- repaired candidate count: `3`
- repaired pass count: `3`
- source max interval: `62`
- repaired max interval: `11`
- target max interval: `12`
- interval reduction: `51`
- required interval reduction min: `50`
- source dead-air max: `0.0000`
- repaired dead-air max: `0.0000`
- min repaired unique pitch count: `22`
- max pitch changed ratio: `0.7174`

## Repair Config

- strategy: `pitch_class_octave_contour_fold`
- preferred pitch range: `48`-`88`
- max adjacent interval: `12`
- min unique pitch count: `8`

## Guardrails

- target max interval: `12`
- target dead-air max: `0.3500`
- max simultaneous notes: `1`
- source max added-note ratio: `0.9167`
- added-note ratio review required: `true`

## Repaired MIDI

- rank `1` sample `1`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/midi/rank_01_sample_01_pitch_contour_repair.mid`, max interval `62` -> `9`, unique pitch `31` -> `23`, pitch changed ratio `0.6957`, pass `true`
- rank `2` sample `2`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/midi/rank_02_sample_02_pitch_contour_repair.mid`, max interval `51` -> `11`, unique pitch `30` -> `26`, pitch changed ratio `0.6522`, pass `true`
- rank `3` sample `3`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/midi/rank_03_sample_03_pitch_contour_repair.mid`, max interval `35` -> `6`, unique pitch `31` -> `22`, pitch changed ratio `0.7174`, pass `true`

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- audio rendered quality claimed: `false`
