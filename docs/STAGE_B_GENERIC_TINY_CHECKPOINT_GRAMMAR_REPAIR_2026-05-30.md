# Stage B Generic Tiny Checkpoint Grammar Repair

## Summary

- boundary: `stage_b_generic_tiny_checkpoint_grammar_repair`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_repeatability_probe`
- grammar repair passed: `true`
- raw generation quality claimed: `false`
- constrained generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Comparison

- baseline valid/strict/grammar: `0/0/0`
- repair valid/strict/grammar: `2/2/2`
- grammar gate delta: `2`
- valid sample delta: `2`
- strict valid sample delta: `2`
- repair collapse warning sample rate: `0.0`
- repair avg postprocess removal ratio: `0.125`
- repair avg onset coverage ratio: `0.1875`
- repair avg sustained coverage ratio: `0.375`

## Baseline Failure Reasons

- `note count too low: 4 < 6`: `1`
- `note count too low: 3 < 6; collapse=single_pitch,single_position`: `1`

## Repair Failure Reasons

- none

## Not Proven

- `unconstrained_raw_generation_quality`
- `generic_base_multi_seed_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
