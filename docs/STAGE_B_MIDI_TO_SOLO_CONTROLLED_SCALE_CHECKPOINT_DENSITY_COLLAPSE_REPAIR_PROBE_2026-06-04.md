# Stage B MIDI-to-Solo Controlled Scale Checkpoint Density Collapse Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision`
- density/collapse target supported: `true`
- strict gate recovered: `false`
- MIDI-to-solo musical quality claimed: `false`

## Repair Result

- sample count: `3`
- valid / strict / grammar: `0` / `0` / `3`
- note-count / dead-air failure count: `0` / `3`
- collapse warning count / rate: `0` / `0.0`
- avg onset / sustained coverage ratio: `0.4583333333333333` / `0.71875`
- avg / max postprocess removal ratio: `0.22916666666666666` / `0.3125`

## Comparison

- note count failure delta: `3`
- collapse warning delta: `3`
- postprocess removal delta: `0.5798761423761424`
- onset / sustained coverage delta: `0.375` / `0.5520833333333334`

## Failure Reasons

- `dead-air ratio too high: 0.800 >= 0.800`: `1`
- `dead-air ratio too high: 0.917 >= 0.800`: `2`

## Not Proven

- `strict_gate_recovered`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
