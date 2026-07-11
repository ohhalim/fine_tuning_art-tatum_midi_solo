# Stage B MIDI-to-Solo Candidate Failure Labeling

## Summary

- boundary: `stage_b_midi_to_solo_candidate_failure_labeling`
- source boundary: `stage_b_midi_to_solo_quality_rubric_baseline`
- next boundary: `stage_b_midi_to_solo_targeted_quality_repair_sweep`
- selected target: `targeted_quality_repair_sweep`
- candidate count: `6`
- failed candidate count: `6`
- outside-soloing repair evidence ready: `true`
- outside-soloing repair WAV count: `6`
- outside-soloing pitch-role risk after: `0`
- outside-soloing not evaluable count: `6`
- targeted quality repair sweep ready: `true`

## Failure Counts

- `dead_air_or_density_gap`: `1`
- `phrase_shape_missing_tension_release`: `2`
- `rhythmic_monotony`: `3`
- `songlike_melody_not_soloing`: `6`

## Not Evaluable

- `outside_soloing_without_context`: `6`
- `weak_chord_tone_landing`: `6`

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Next

- `Stage B MIDI-to-solo targeted quality repair sweep`
