# Stage B Generic Base Scale Checkpoint Density Coverage Repair Probe

## Summary

- boundary: `stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe`
- next boundary: `stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision`
- density/coverage target qualified: `true`
- raw generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Baseline

- sample count: `3`
- valid / strict / grammar gate: `0` / `0` / `0`
- note count failure count: `3`
- avg onset / sustained coverage: `0.0625` / `0.09375`

## Repair

- sample count: `3`
- valid / strict / grammar gate: `1` / `1` / `3`
- note count failure count: `0`
- avg onset / sustained coverage: `0.16666666666666666` / `0.6354166666666666`
- max longest sustained empty run steps: `7`

## Delta

- note count failure delta: `3`
- onset coverage delta: `0.10416666666666666`
- sustained coverage delta: `0.5416666666666666`

## Remaining Failure Reasons

- `too many long notes: 0.333 > 0.250`: `2`

## Not Proven

- `raw_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
