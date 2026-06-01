# Stage B Generic Model-Core Training Data Plan

## Summary

- boundary: `stage_b_generic_model_core_training_data_plan`
- repair loop status: `stopped`
- tiny checkpoint role: `diagnostic_only`
- next method: `generic_manifest_full_window_preparation_then_training`
- generic train / val files: `2433` / `270`
- Brad files excluded from generic base: `true`
- full window preparation executed: `false`
- full training executed: `false`
- broad trained model quality claimed: `false`
- next boundary: `stage_b_generic_full_manifest_window_preparation`

## Evidence

- manifest generic train / val: `2433` / `270`
- manifest Brad split: `47` / `11` / `14`
- window smoke selected train / val files: `6` / `3`
- window smoke token max / vocab: `544` / `547`
- tiny training selected train / val records: `32` / `8`
- tiny training best validation loss: `6.1427`

## Execution Order

| step | name | goal | stop condition |
|---:|---|---|---|
| 1 | `full_generic_manifest_window_preparation` | convert full non-Brad generic train/val manifests to Stage B windows | token ids exceed vocab or train/val boundary changes |
| 2 | `full_window_token_guard` | record train/val window counts, non-empty token counts, max token id, vocab fit | empty validation split or vocab overflow |
| 3 | `generic_base_training_scale_smoke` | run controlled larger-than-tiny training with validation loss and checkpoint metadata | training returncode nonzero or validation artifact missing |
| 4 | `generic_base_generation_probe` | evaluate raw model output before constrained rescue | raw generation fails structural gate; record as model-core failure |
| 5 | `review_package_and_audio_boundary` | render only structurally reviewable candidates for listening review | no candidate passes objective review gate |

## Not Proven

- `full_generic_window_preparation`
- `full_generic_training_run`
- `broad_trained_model_quality`
- `brad_style_adaptation`
- `production_ready_improviser`
