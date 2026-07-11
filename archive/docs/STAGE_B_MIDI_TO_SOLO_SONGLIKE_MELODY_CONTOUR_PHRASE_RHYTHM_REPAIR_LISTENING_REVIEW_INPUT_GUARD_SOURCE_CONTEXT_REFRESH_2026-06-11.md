# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Listening Review Input Guard Source Context Refresh

## Summary

- boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard`
- schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard_v5`
- source boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package`
- source listening review package schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package_v5`
- source audio package schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_v5`
- source phrase/rhythm repair sweep schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep_v5`
- source follow-up schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision_v5`
- source objective next schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_objective_next_v5`
- source repair sweep schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_sweep_v5`
- source repair listening review input guard schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_input_guard_v5`
- source repair listening review package schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package_v5`
- source repair audio package schema version: `stage_b_midi_to_solo_songlike_melody_contour_repair_audio_package_v5`
- next boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision`
- review item count: `6`
- required input field count: `4`
- validated review input present: `false`
- preference fill allowed: `false`
- technical WAV validation: `true`
- rendered audio file count: `6`
- duration range: `18.871s-19.000s`
- failure label delta: `3`
- phrase/rhythm failure count: `4 -> 1`
- phrase/rhythm failure delta: `3`
- source outside-soloing repair evidence ready: `true`
- objective source outside-soloing source context preserved: `true`
- objective source outside-soloing schema context preserved: `true`
- objective source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- objective source outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- objective source outside-soloing current repair pitch-role risk after / delta: `0` / `2`
- source outside-soloing source context preserved: `true`
- source outside-soloing schema context preserved: `true`
- source outside-soloing objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing source context preserved: `true`
- source outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- source outside-soloing source repair targeted: `false`
- source outside-soloing source residual risk preserved: `true`
- source outside-soloing repair pitch-role risk count after: `0`
- source outside-soloing current repair pitch-role risk delta: `2`
- source/repaired outside-soloing not evaluable count: `6/6`
- audio review required: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- reason: `listening review input pending; preference fill blocked`
- next recommended issue: `Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair objective-only next decision source-context refresh`

## Required Input Fields

- `candidate_index`
- `listening_status`
- `preference`
- `issue_notes`

## Repaired Not Evaluable Counts

- `outside_soloing_without_context`: `6`
- `weak_chord_tone_landing`: `6`

## Review WAV Paths

- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_01_cli_repaired_midi_rank_01_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_02_cli_repaired_midi_rank_02_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_03_cli_repaired_midi_rank_03_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_04_changed_ratio_repair_rank_04_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_05_changed_ratio_repair_rank_05_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package_source_context_refresh/audio/candidate_06_changed_ratio_repair_rank_06_phrase_rhythm_repair.wav`

## Claim Boundary

- `listening_review_completed`
- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
