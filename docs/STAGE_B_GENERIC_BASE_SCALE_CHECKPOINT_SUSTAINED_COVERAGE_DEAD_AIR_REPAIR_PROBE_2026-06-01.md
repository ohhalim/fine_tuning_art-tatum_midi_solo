# Stage B Generic Base Scale Checkpoint Sustained Coverage Dead-Air Repair Probe

## Summary

- boundary: `stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe`
- next boundary: `stage_b_generic_base_scale_checkpoint_objective_gate_consolidation`
- sustained coverage dead-air target qualified: `true`
- raw generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Baseline

- sample count: `3`
- valid / strict / grammar gate: `2` / `2` / `3`
- dead-air / long-note failure count: `1` / `0`
- avg onset / sustained coverage: `0.1875` / `0.3645833333333333`
- max longest sustained empty run steps: `8`

## Repair

- constrained note groups per bar: `8`
- sample count: `3`
- valid / strict / grammar gate: `3` / `3` / `3`
- dead-air / long-note failure count: `0` / `0`
- avg onset / sustained coverage: `0.3854166666666667` / `0.6354166666666666`
- max longest sustained empty run steps: `4`

## Delta

- dead-air failure delta: `1`
- valid / strict sample delta: `1` / `1`
- onset / sustained coverage delta: `0.19791666666666669` / `0.2708333333333333`
- long-note failure reintroduced: `false`

## Remaining Failure Reasons

- none

## Not Proven

- `raw_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
