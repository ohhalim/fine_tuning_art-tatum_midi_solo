# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Remaining Blocker Decision

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision`
- decision: `select_dead_air_repair_probe`
- selected target: `dead_air_sustained_coverage_repair`
- remaining blocker: `dead_air_sustained_coverage`
- audio review selected: `false`
- training scale change selected: `false`
- MIDI-to-solo musical quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe`

## Evidence

- sample count: `3`
- valid / strict / grammar gate sample count: `0` / `0` / `3`
- note-count failure count: `0`
- collapse warning sample count: `0`
- dead-air failure count: `3`
- avg postprocess removal ratio: `0.22916666666666666`
- avg onset / sustained coverage ratio: `0.4583333333333333` / `0.71875`
- max longest sustained empty run steps: `2`

## Failure Reasons

- `dead-air ratio too high: 0.800 >= 0.800`: `1`
- `dead-air ratio too high: 0.917 >= 0.800`: `2`

## Not Proven

- `strict_gate_recovered`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
