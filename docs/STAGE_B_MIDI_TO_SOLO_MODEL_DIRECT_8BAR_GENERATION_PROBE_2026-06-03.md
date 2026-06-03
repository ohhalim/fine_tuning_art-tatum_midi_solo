# Stage B MIDI-to-Solo Model-Direct 8-Bar Generation Probe

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_8bar_generation_probe`
- next boundary: `stage_b_midi_to_solo_model_direct_monophonic_overlap_repair`
- generation source: `model_checkpoint_direct_constrained`
- direct generated MIDI written: `true`
- direct generation grammar gate passed: `true`
- direct generation review gate passed: `false`
- model-direct generation quality claimed: `false`

## Context

- target bars: `8`
- BPM: `120`
- chord progression: `Cmaj7, F7, G7, Cmaj7, Cmaj7, Cmaj7, Cmaj7, Cmaj7`
- low-confidence chord bars: `4`

## Generation

- max sequence: `160`
- note groups per bar: `3`
- sample count: `3`
- grammar gate sample count: `3`
- valid sample count: `0`
- strict valid sample count: `0`
- min pre-postprocess note groups: `24`
- min postprocess note count: `10`
- max postprocess note count: `12`
- avg postprocess removal ratio: `0.5416666666666666`
- collapse warning sample rate: `1.0`

## MIDI Paths

- `outputs/stage_b_midi_to_solo_model_direct_8bar_generation_probe/harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe/generation_probe/model_direct_8bar/samples/stage_b_sample_1.mid`
- `outputs/stage_b_midi_to_solo_model_direct_8bar_generation_probe/harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe/generation_probe/model_direct_8bar/samples/stage_b_sample_2.mid`
- `outputs/stage_b_midi_to_solo_model_direct_8bar_generation_probe/harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe/generation_probe/model_direct_8bar/samples/stage_b_sample_3.mid`

## Not Proven

- `model_checkpoint_direct_8bar_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
