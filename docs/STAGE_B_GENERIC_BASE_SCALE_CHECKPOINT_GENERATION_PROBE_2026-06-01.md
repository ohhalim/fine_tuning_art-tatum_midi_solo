# Stage B Generic Base Scale Checkpoint Generation Probe

## Summary

- boundary: `stage_b_generic_base_scale_checkpoint_generation_probe`
- next boundary: `stage_b_generic_base_scale_checkpoint_grammar_representation_decision`
- generation path executable: `true`
- raw generation quality ready: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Training Source

- source tokenized train / val records: `154136` / `21845`
- selected train / val records: `128` / `32`
- best validation loss: `5.9031`
- checkpoint count: `1`

## Generation

- command returncode: `0`
- sample count: `3`
- valid sample count: `0`
- strict valid sample count: `0`
- grammar gate sample count: `0`
- collapse warning sample rate: `0.0`
- avg onset coverage ratio: `0.0625`
- avg sustained coverage ratio: `0.09375`
- max longest sustained empty run steps: `25`

## Failure Reasons

- `note count too low: 4 < 6`: `1`
- `note count too low: 3 < 6`: `1`
- `note count too low: 2 < 6`: `1`

## Not Proven

- `full_generic_training_run`
- `generic_base_generation_quality`
- `generic_base_multi_seed_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
