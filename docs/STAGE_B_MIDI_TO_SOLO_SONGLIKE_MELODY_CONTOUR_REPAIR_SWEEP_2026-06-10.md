# Stage B MIDI-to-Solo Songlike Melody Contour Repair Sweep

## Summary

- boundary: `stage_b_midi_to_solo_songlike_melody_contour_repair_sweep`
- source boundary: `stage_b_midi_to_solo_targeted_quality_repair_followup_decision`
- next boundary: `stage_b_midi_to_solo_songlike_melody_contour_repair_audio_package`
- selected target: `songlike_melody_contour_repair_audio_package`
- candidate count: `6`
- total failure labels: `8 -> 4`
- failure label delta: `4`
- songlike failure count: `5 -> 0`
- songlike failure delta: `5`
- improved candidate count: `4`
- technical regression count: `0`
- source outside-soloing repair evidence ready: `true`
- source outside-soloing repair pitch-role risk after: `0`
- source outside-soloing not evaluable count: `6`
- repaired outside-soloing not evaluable count: `6`
- target supported: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Repaired Failure Counts

- `phrase_shape_missing_tension_release`: `2`
- `rhythmic_monotony`: `2`

## Repaired Not Evaluable Counts

- `outside_soloing_without_context`: `6`
- `weak_chord_tone_landing`: `6`

## MIDI Files

- rank `1`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/01_songlike_melody_contour_repair.mid`
- rank `2`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/02_songlike_melody_contour_repair.mid`
- rank `3`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/03_songlike_melody_contour_repair.mid`
- rank `4`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/04_songlike_melody_contour_repair.mid`
- rank `5`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/05_songlike_melody_contour_repair.mid`
- rank `6`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep/midi/06_songlike_melody_contour_repair.mid`

## Decision

- reason: `songlike contour repair sweep completed without musical quality claim`
- auto progress allowed: `true`
- critical user input required: `false`
- next recommended issue: `Stage B MIDI-to-solo songlike melody contour repair audio package`

## Claim Boundary

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `outside_soloing_without_context`
- `weak_chord_tone_landing`
- `broad_trained_model_quality`
