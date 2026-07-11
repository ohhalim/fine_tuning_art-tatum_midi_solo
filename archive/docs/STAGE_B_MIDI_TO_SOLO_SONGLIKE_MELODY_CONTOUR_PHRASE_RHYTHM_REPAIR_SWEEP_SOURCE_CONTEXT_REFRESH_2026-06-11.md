# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Sweep

## Summary

- boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep`
- source boundary: `stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision`
- schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_v5`
- source follow-up schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision_v5`
- source objective next schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_objective_next_v5`
- source repair sweep schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_sweep_v5`
- source input guard schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_input_guard_v5`
- source listening review package schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package_v5`
- source audio package schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_audio_package_v5`
- next boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package`
- selected target: `songlike_melody_contour_phrase_rhythm_repair_audio_package`
- candidate count: `6`
- total failure labels: `4 -> 1`
- failure label delta: `3`
- phrase/rhythm failure count: `4 -> 1`
- phrase/rhythm failure delta: `3`
- improved candidate count: `2`
- technical regression count: `0`
- source outside-soloing repair evidence ready: `true`
- objective source outside-soloing source context preserved: `true`
- objective source outside-soloing schema context preserved: `true`
- objective source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- source outside-soloing source context preserved: `true`
- source outside-soloing schema context preserved: `true`
- source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing source context preserved: `true`
- source outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- source outside-soloing source repair targeted: `false`
- source outside-soloing source residual risk preserved: `true`
- source outside-soloing repair pitch-role risk after: `0`
- source outside-soloing current repair pitch-role risk delta: `2`
- source outside-soloing not evaluable count: `6`
- repaired outside-soloing not evaluable count: `6`
- target supported: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Repaired Failure Counts

- `rhythmic_monotony`: `1`

## Repaired Not Evaluable Counts

- `outside_soloing_without_context`: `6`
- `weak_chord_tone_landing`: `6`

## MIDI Files

- rank `1`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/01_songlike_melody_contour_phrase_rhythm_repair.mid`
- rank `2`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/02_songlike_melody_contour_phrase_rhythm_repair.mid`
- rank `3`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/03_songlike_melody_contour_phrase_rhythm_repair.mid`
- rank `4`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/04_songlike_melody_contour_phrase_rhythm_repair.mid`
- rank `5`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/05_songlike_melody_contour_phrase_rhythm_repair.mid`
- rank `6`: `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_source_context_refresh/midi/06_songlike_melody_contour_phrase_rhythm_repair.mid`

## Decision

- reason: `phrase/rhythm repair sweep completed without musical quality claim`
- auto progress allowed: `true`
- critical user input required: `false`
- next recommended issue: `Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair audio package source-context refresh`

## Claim Boundary

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `outside_soloing_without_context`
- `weak_chord_tone_landing`
- `broad_trained_model_quality`
