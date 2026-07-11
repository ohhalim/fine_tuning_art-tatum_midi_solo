# Stage B MIDI-to-Solo Quality Rubric Baseline Source Context Refresh

## Summary

- boundary: `stage_b_midi_to_solo_quality_rubric_baseline`
- source boundary: `stage_b_midi_to_solo_post_mvp_quality_iteration_plan`
- schema version: `stage_b_midi_to_solo_quality_rubric_baseline_v4`
- source post-MVP plan schema version: `stage_b_midi_to_solo_post_mvp_quality_iteration_plan_v4`
- source final status schema version: `stage_b_midi_to_solo_final_status_audit_v4`
- source delivery package schema version: `stage_b_midi_to_solo_mvp_delivery_package_v4`
- source listening gap schema version: `stage_b_midi_to_solo_listening_review_quality_gap_v4`
- source quality gap schema version: `stage_b_midi_to_solo_quality_gap_decision_v4`
- source current evidence schema version: `stage_b_midi_to_solo_mvp_current_evidence_consolidation_v4`
- next boundary: `stage_b_midi_to_solo_candidate_failure_labeling`
- selected target: `candidate_failure_labeling`
- rubric item count: `8`
- candidate failure labeling ready: `true`
- outside-soloing repair evidence ready: `true`
- outside-soloing repair source context preserved: `true`
- outside-soloing repair schema context preserved: `true`
- outside-soloing repair objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing source context preserved: `true`
- outside-soloing repair WAV count: `6`
- outside-soloing source objective pitch-role risk: `5`
- outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk after / delta: `0` / `2`
- follow-up objective source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- follow-up repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- bridge repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`

## Rubric Items

- `sparse_or_empty_output`: note_count < 12 or active_bar_count < 2
- `dead_air_or_density_gap`: dead_air_ratio > 0.35 or empty_bar_count > 0
- `rhythmic_monotony`: duration_most_common_ratio >= 0.40 or ioi_most_common_ratio >= 0.40 or note_count_per_bar_most_common_ratio >= 0.95
- `songlike_melody_not_soloing`: four_notes_per_bar_template or four_bar_rhythm_cycle_repeated or shared_rhythm_signature_count >= 3
- `outside_soloing_without_context`: outside_pitch_run_length >= 4 or avoid_tone_landing_count > 0 when chord_context_available
- `weak_chord_tone_landing`: cadence_landing_chord_tone == false or strong_beat_chord_tone_ratio < 0.40
- `phrase_shape_missing_tension_release`: contour_turn_count == 0 or cadence_resolution_present == false
- `technical_gate_regression`: grammar_valid == false or strict_valid == false or max_simultaneous_notes > 1

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- broad trained model quality claimed: `false`

## Next

- `Stage B MIDI-to-solo candidate failure labeling source-context refresh`
