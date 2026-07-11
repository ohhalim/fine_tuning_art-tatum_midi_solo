# Stage B Generic Base Manifest Contract

## Summary

- boundary: `stage_b_generic_base_manifest_contract`
- next boundary: `stage_b_generic_stage_b_window_prepare_smoke`
- manifest contract ready: `true`
- stage_b window prepare smoke ready: `true`
- broad training execution ready: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Split Counts

- generic_jazz_train: `2433`
- generic_jazz_val: `270`
- expected non-Brad candidates: `2703`
- actual non-Brad split count: `2703`
- brad_adaptation_train: `47`
- brad_adaptation_val: `11`
- brad_test_holdout: `14`
- expected Brad candidates: `72`
- actual Brad split count: `72`

## Guards

- generic Brad leak count: `0`
- Brad non-Brad leak count: `0`
- overlap path count: `0`
- duplicate exact hash group count: `0`
- duplicate exact file count: `0`

## Not Proven

- `stage_b_generic_window_prepare_smoke`
- `generic_base_training_run`
- `generic_base_multi_seed_quality`
- `brad_style_adaptation`
