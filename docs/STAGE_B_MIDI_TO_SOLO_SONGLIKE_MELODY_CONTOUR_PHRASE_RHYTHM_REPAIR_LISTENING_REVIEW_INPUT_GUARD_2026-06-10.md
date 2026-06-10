# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Listening Review Input Guard

## Summary

- boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard`
- source boundary: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package`
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
- source outside-soloing repair pitch-role risk count after: `0`
- source/repaired outside-soloing not evaluable count: `6/6`
- audio review required: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- reason: `listening review input pending; preference fill blocked`
- next recommended issue: `Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair objective-only next decision`

## Required Input Fields

- `candidate_index`
- `listening_status`
- `preference`
- `issue_notes`

## Repaired Not Evaluable Counts

- `outside_soloing_without_context`: `6`
- `weak_chord_tone_landing`: `6`

## Review WAV Paths

- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_01_cli_repaired_midi_rank_01_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_02_cli_repaired_midi_rank_02_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_03_cli_repaired_midi_rank_03_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_04_changed_ratio_repair_rank_04_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_05_changed_ratio_repair_rank_05_phrase_rhythm_repair.wav`
- `outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/harness_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_audio_package/audio/candidate_06_changed_ratio_repair_rank_06_phrase_rhythm_repair.wav`

## Claim Boundary

- `listening_review_completed`
- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
