# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Density Grammar Collapse Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe`
- density/grammar/collapse target supported: `true`
- strict gate recovered: `true`
- MIDI-to-solo musical quality claimed: `false`

## Repair Result

- sample count: `3`
- valid / strict / grammar: `1` / `1` / `3`
- note-count / grammar / dead-air failure count: `0` / `0` / `2`
- collapse warning count / rate: `0` / `0.0`
- avg onset / sustained coverage ratio: `0.46875` / `0.6145833333333334`
- avg / max postprocess removal ratio: `0.1875` / `0.25`

## Comparison

- note count failure delta: `3`
- grammar failure delta: `1`
- collapse warning delta: `3`
- postprocess removal delta: `0.603409090909091`
- onset / sustained coverage delta: `0.3541666666666667` / `0.46875`

## Failure Reasons

- `dead-air ratio too high: 0.833 >= 0.800`: `1`
- `dead-air ratio too high: 0.818 >= 0.800`: `1`

## Not Proven

- `repeatability`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
