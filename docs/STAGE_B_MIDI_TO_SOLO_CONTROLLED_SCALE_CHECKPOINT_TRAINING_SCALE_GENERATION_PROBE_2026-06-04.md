# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Generation Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision`
- train / val records: `2048` / `512`
- best validation loss: `3.0396`
- sample count: `3`
- valid / strict / grammar: `0` / `0` / `2`
- collapse warning sample count / rate: `3` / `1.0`
- avg onset / sustained coverage ratio: `0.11458333333333333` / `0.14583333333333334`
- avg / max postprocess removal ratio: `0.790909090909091` / `0.8`
- raw generation quality ready: `false`
- MIDI-to-solo musical quality claimed: `false`

## Failure Reasons

- `note count too low: 4 < 6; collapse=postprocess_removed_majority`: `1`
- `note count too low: 5 < 6; collapse=postprocess_removed_majority`: `2`

## Not Proven

- `selected_scale_checkpoint_generation_repeatability`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
