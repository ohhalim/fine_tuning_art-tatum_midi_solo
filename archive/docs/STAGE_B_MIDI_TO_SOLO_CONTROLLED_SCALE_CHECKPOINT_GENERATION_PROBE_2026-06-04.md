# Stage B MIDI-to-Solo Controlled Scale Checkpoint Generation Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision`
- train / val records: `512` / `128`
- best validation loss: `5.1061`
- sample count: `3`
- valid / strict / grammar: `0` / `0` / `3`
- collapse warning sample count / rate: `3` / `1.0`
- avg onset / sustained coverage ratio: `0.08333333333333333` / `0.16666666666666666`
- max longest sustained empty run steps: `32`
- avg / max postprocess removal ratio: `0.809042809042809` / `0.8636363636363636`
- raw generation quality ready: `false`
- MIDI-to-solo musical quality claimed: `false`

## Failure Reasons

- `note count too low: 4 < 6; collapse=postprocess_removed_majority`: `1`
- `note count too low: 3 < 6; collapse=repeated_position_pitch,postprocess_removed_majority`: `1`
- `note count too low: 3 < 6; collapse=postprocess_removed_majority`: `1`

## Not Proven

- `checkpoint_generation_repeatability`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
