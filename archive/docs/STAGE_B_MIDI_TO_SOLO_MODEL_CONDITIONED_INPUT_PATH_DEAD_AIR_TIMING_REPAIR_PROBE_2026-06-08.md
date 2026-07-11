# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package`
- repair passed: `true`
- source candidate count: `3`
- repaired candidate count: `3`
- repaired pass count: `3`
- source dead-air max: `0.6522`
- repaired dead-air max: `0.0000`
- dead-air gain max: `0.6522`
- max added-note ratio: `0.9167`
- max postprocess removal ratio: `0.0000`
- max repaired simultaneous notes: `1`
- max repaired interval: `62`

## Repair Config

- strategy: `timing_gap_fill_and_duration_compaction`
- dead-air threshold seconds: `0.5000`
- max start gap seconds: `0.4900`
- fill note duration seconds: `0.1800`
- preferred fill pitch range: `48`-`88`

## Guardrails

- target dead-air max: `0.3500`
- required dead-air gain min: `0.3022`
- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`
- max postprocess removal ratio: `0.2500`

## Repaired MIDI

- rank `1` sample `1`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/midi/rank_01_sample_01_dead_air_timing_repair.mid`, dead-air `0.6522` -> `0.0000`, notes `24` -> `46`, added ratio `0.9167`, pass `true`
- rank `2` sample `2`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/midi/rank_02_sample_02_dead_air_timing_repair.mid`, dead-air `0.6522` -> `0.0000`, notes `24` -> `46`, added ratio `0.9167`, pass `true`
- rank `3` sample `3`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/midi/rank_03_sample_03_dead_air_timing_repair.mid`, dead-air `0.6522` -> `0.0000`, notes `24` -> `46`, added ratio `0.9167`, pass `true`

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- audio-rendered quality claimed: `false`
- model checkpoint generation quality claimed: `false`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- next recommended issue: `Stage B MIDI-to-solo model-conditioned input path dead-air timing repair audio package`
