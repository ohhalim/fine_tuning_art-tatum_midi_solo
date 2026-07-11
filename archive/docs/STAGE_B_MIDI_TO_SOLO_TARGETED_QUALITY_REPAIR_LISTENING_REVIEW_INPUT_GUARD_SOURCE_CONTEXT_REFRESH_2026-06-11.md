# Stage B MIDI-to-Solo Targeted Quality Repair Listening Review Input Guard Source Context Refresh

## Summary

- boundary: `stage_b_midi_to_solo_targeted_quality_repair_listening_review_input_guard`
- source boundary: `stage_b_midi_to_solo_targeted_quality_repair_listening_review_package`
- next boundary: `stage_b_midi_to_solo_targeted_quality_repair_objective_only_next_decision`
- schema version: `stage_b_midi_to_solo_targeted_quality_repair_listening_review_input_guard_v5`
- source targeted quality repair listening review package schema version: `stage_b_midi_to_solo_targeted_quality_repair_listening_review_package_v5`
- source targeted quality repair audio package schema version: `stage_b_midi_to_solo_targeted_quality_repair_audio_package_v5`
- source targeted quality repair sweep schema version: `stage_b_midi_to_solo_targeted_quality_repair_sweep_v4`
- source candidate failure labeling schema version: `stage_b_midi_to_solo_candidate_failure_labeling_v4`
- source quality rubric schema version: `stage_b_midi_to_solo_quality_rubric_baseline_v4`
- source post-MVP plan schema version: `stage_b_midi_to_solo_post_mvp_quality_iteration_plan_v4`
- source final status schema version: `stage_b_midi_to_solo_final_status_audit_v4`
- source delivery package schema version: `stage_b_midi_to_solo_mvp_delivery_package_v4`
- source listening gap schema version: `stage_b_midi_to_solo_listening_review_quality_gap_v4`
- source quality gap schema version: `stage_b_midi_to_solo_quality_gap_decision_v4`
- source current evidence schema version: `stage_b_midi_to_solo_mvp_current_evidence_consolidation_v4`
- review item count: `6`
- required input field count: `4`
- validated review input present: `false`
- preference fill allowed: `false`
- technical WAV validation: `true`
- rendered audio file count: `6`
- duration range: `18.422s-18.984s`
- failure label delta: `4`
- source outside-soloing repair evidence ready: `true`
- source outside-soloing repair source context preserved: `true`
- source outside-soloing repair schema context preserved: `true`
- source outside-soloing repair objective schema version: `stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing source context preserved: `true`
- source outside-soloing repair WAV count: `6`
- source outside-soloing source objective pitch-role risk: `5`
- source outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- source outside-soloing source repair targeted: `false`
- source outside-soloing source residual risk preserved: `true`
- source outside-soloing current repair pitch-role risk after / delta: `0` / `2`
- follow-up objective source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `0` / `2`
- follow-up repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `0` / `2`
- bridge repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `0` / `2`
- source outside-soloing not evaluable count: `6`
- repaired outside-soloing not evaluable count: `6`
- audio review required: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- reason: `listening review input pending; preference fill blocked`
- next recommended issue: `Stage B MIDI-to-solo targeted quality repair objective-only next decision source-context refresh`

## Required Input Fields

- `candidate_index`
- `listening_status`
- `preference`
- `issue_notes`

## Review WAV Paths

- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_01_cli_repaired_midi_rank_01_targeted_quality_repair.wav`
- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_02_cli_repaired_midi_rank_02_targeted_quality_repair.wav`
- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_03_cli_repaired_midi_rank_03_targeted_quality_repair.wav`
- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_04_changed_ratio_repair_rank_01_targeted_quality_repair.wav`
- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_05_changed_ratio_repair_rank_02_targeted_quality_repair.wav`
- `outputs/stage_b_midi_to_solo_targeted_quality_repair_audio_package/harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package_source_context_refresh/audio/candidate_06_changed_ratio_repair_rank_03_targeted_quality_repair.wav`

## Claim Boundary

- `listening_review_completed`
- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `audio_rendered_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
