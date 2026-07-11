# Stage B MIDI-to-Solo Phrase/Rhythm Chord-Context Pitch-Role Bridge

## Summary

- boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge`
- schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge_v5`
- source boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision`
- repair sweep boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep`
- source follow-up schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision_v5`
- source objective next schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_next_v5`
- source repair sweep schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_v5`
- source input guard schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard_v5`
- source listening review package schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package_v5`
- source audio package schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_v5`
- next boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision`
- selected target: `songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision`
- chord progression: `Cm7,Fm7,Bb7,Ebmaj7`
- context source: `fallback_default_harness_chords`
- candidate count: `6`
- chord context available count: `6`
- pitch-role metrics defined count: `6`
- not evaluable count: `12 -> 0`
- follow-up objective source/repaired outside-soloing not evaluable count: `6/6`
- follow-up repair sweep source/repaired outside-soloing not evaluable count: `6/6`
- bridge repair sweep source/repaired outside-soloing not evaluable count: `6/6`
- follow-up objective source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up objective source outside-soloing source pitch-role risk delta: `3`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up objective source outside-soloing schema context preserved: `true`
- follow-up objective source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up objective source outside-soloing source targeted: `false`
- follow-up objective source outside-soloing source residual risk preserved: `true`
- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- follow-up repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up repair sweep source outside-soloing source pitch-role risk delta: `3`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing schema context preserved: `true`
- follow-up repair sweep source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up repair sweep source outside-soloing source targeted: `false`
- follow-up repair sweep source outside-soloing source residual risk preserved: `true`
- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- bridge repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- bridge repair sweep source outside-soloing source pitch-role risk delta: `3`
- bridge repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing schema context preserved: `true`
- bridge repair sweep source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- bridge repair sweep source outside-soloing source targeted: `false`
- bridge repair sweep source outside-soloing source residual risk preserved: `true`
- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- min chord-tone ratio: `0.216`
- max outside ratio: `0.027`
- max non-chord run: `5`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Bridge Flags

- `outside_soloing_pitch_role_risk`: `5`
- `weak_chord_tone_landing_risk`: `6`

## Candidates

| rank | chord-tone | tension | approach | outside | strong beat chord-tone | final role | flags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.297 | 0.270 | 0.432 | 0.000 | 0.545 | `approach` | `weak_chord_tone_landing_risk` |
| 2 | 0.457 | 0.086 | 0.457 | 0.000 | 0.300 | `approach` | `outside_soloing_pitch_role_risk,weak_chord_tone_landing_risk` |
| 3 | 0.378 | 0.108 | 0.486 | 0.027 | 0.400 | `approach` | `outside_soloing_pitch_role_risk,weak_chord_tone_landing_risk` |
| 4 | 0.216 | 0.243 | 0.541 | 0.000 | 0.333 | `guide` | `outside_soloing_pitch_role_risk,weak_chord_tone_landing_risk` |
| 5 | 0.237 | 0.211 | 0.526 | 0.026 | 0.000 | `approach` | `outside_soloing_pitch_role_risk,weak_chord_tone_landing_risk` |
| 6 | 0.324 | 0.270 | 0.405 | 0.000 | 0.167 | `approach` | `outside_soloing_pitch_role_risk,weak_chord_tone_landing_risk` |

## Decision

- reason: `chord-context bridge produced pitch-role metrics without quality claim`
- auto progress allowed: `true`
- critical user input required: `false`
- next recommended issue: `Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role objective decision source-context refresh`

## Claim Boundary

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
