# Stage B Generic Base Training Scale Smoke

## Summary

- boundary: `stage_b_generic_base_training_scale_smoke`
- next boundary: `stage_b_generic_base_scale_checkpoint_generation_probe`
- training scale smoke passed: `true`
- scale training smoke executed: `true`
- full generic training executed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Source Window

- train / val manifest files: `2433` / `270`
- source tokenized train / val records: `154136` / `21845`
- source max token id / vocab size: `544` / `547`

## Input

- selected train / val records: `128` / `32`
- min train / val records: `64` / `16`
- token files: `160`
- max token id / vocab size: `544` / `547`
- fits vocab: `true`

## Training

- returncode: `0`
- best validation loss: `5.9031`
- checkpoint count: `1`
- lora weights exists: `true`

## Not Proven

- `full_generic_training_run`
- `generic_base_generation_quality`
- `generic_base_multi_seed_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
