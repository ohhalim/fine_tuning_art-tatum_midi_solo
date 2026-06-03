# Stage B MIDI-to-Solo Model-Direct Monophonic Overlap Repair

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_monophonic_overlap_repair`
- next boundary: `stage_b_midi_to_solo_model_direct_audio_render_package`
- cap duration to next position: `true`
- postprocess removal reduced: `true`
- direct generation review gate passed: `true`
- model-direct generation quality claimed: `false`

## Before / After

- valid sample count: `0` -> `3`
- strict valid sample count: `0` -> `3`
- avg postprocess removal ratio: `0.5416666666666666` -> `0.0`
- collapse warning sample rate: `1.0` -> `0.0`
- min postprocess note count: `10` -> `24`

## Repaired Generation

- sample count: `3`
- grammar gate sample count: `3`
- valid sample count: `3`
- strict valid sample count: `3`
- min postprocess note count: `24`
- max postprocess note count: `24`

## MIDI Paths

- `outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/generation_probe/monophonic_overlap_repair/samples/stage_b_sample_1.mid`
- `outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/generation_probe/monophonic_overlap_repair/samples/stage_b_sample_2.mid`
- `outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/generation_probe/monophonic_overlap_repair/samples/stage_b_sample_3.mid`

## Not Proven

- `model_checkpoint_direct_8bar_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
