# Stage B Generic Base Scale Checkpoint Duration Long-Note Repair Probe

## Summary

- boundary: `stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe`
- next boundary: `stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision`
- duration long-note target qualified: `true`
- raw generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`

## Source Repair

- sample count: `3`
- valid / strict / grammar gate: `1` / `1` / `3`
- long-note failure count: `2`
- avg onset / sustained coverage: `0.16666666666666666` / `0.6354166666666666`
- max longest sustained empty run steps: `7`

## Duration Repair

- sample count: `3`
- valid / strict / grammar gate: `2` / `2` / `3`
- long-note failure count: `0`
- avg onset / sustained coverage: `0.1875` / `0.3645833333333333`
- max longest sustained empty run steps: `8`
- avg duration diversity / most common duration ratio: `0.4583333333333333` / `0.625`

## Delta

- long-note failure delta: `2`
- valid / strict sample delta: `1` / `1`
- onset / sustained coverage delta: `0.020833333333333343` / `-0.2708333333333333`
- coverage regression observed: `true`

## Remaining Failure Reasons

- `dead-air ratio too high: 0.800 >= 0.800`: `1`

## Not Proven

- `raw_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
