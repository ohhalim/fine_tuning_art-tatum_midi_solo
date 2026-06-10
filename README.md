# Jazz Piano MIDI-to-Solo 검증 파이프라인

Symbolic MIDI 기반 jazz piano solo-line 생성 파이프라인.

현재 목표는 완성형 연주 모델이 아니라, 입력 MIDI를 받아 context 추출, 후보 생성, constrained decoding, ranking, MIDI/WAV export, objective review까지 이어지는 model-core MVP 검증이다.

## 현재 상태

- latest functional result: `Issue #898`
- latest MVP completion audit: `Issue #902`
- latest quality gap decision: `Issue #904`
- latest listening review quality gap: `Issue #906`
- latest MVP delivery package: `Issue #908`
- latest README final evidence refresh: `Issue #910`
- latest final status audit: `Issue #912`
- latest post-MVP quality iteration plan: `Issue #914`
- latest quality rubric baseline: `Issue #916`
- latest candidate failure labeling: `Issue #918`
- latest targeted quality repair sweep: `Issue #920`
- latest targeted quality repair audio package: `Issue #922`
- latest README evidence refresh: `Issue #900`
- latest functional boundary: `stage_b_midi_to_solo_targeted_quality_repair_audio_package`
- open issue queue after targeted quality repair audio package source-context refresh merge: `0`
- latest evidence boundary: `stage_b_midi_to_solo_mvp_delivery_package`
- current evidence boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- current MVP evidence support: `true`
- technical model-core MVP completed: `true`
- selected quality gap target: `mvp_delivery_package`
- model-conditioned input path aligned: `false`
- model-conditioned candidate source available: `true`
- model-conditioned audio technical path available: `true`
- model-conditioned input path probe completed: `true`
- model-conditioned ranked input-path export contract matched: `true`
- fallback replacement candidate export ready: `true`
- model-conditioned ranked audio render completed: `true`
- fallback replacement technical path ready: `true`
- fallback replacement ready: `true`
- model-conditioned input path candidate export completed: `true`
- model-conditioned input path audio render completed: `true`
- model-conditioned input path replacement consolidated: `true`
- listening review package required: `true`
- listening review package ready: `true`
- model-conditioned listening review input guard completed: `true`
- model-conditioned preference fill allowed: `false`
- model-conditioned objective-only next decision completed: `true`
- model-conditioned dead-air failure count: `3 / 3`
- model-conditioned dead-air timing repair required: `true`
- model-conditioned dead-air timing repair decision completed: `true`
- model-conditioned target dead-air max: `0.3500`
- model-conditioned required dead-air gain min: `0.3022`
- model-conditioned dead-air timing repair probe completed: `true`
- model-conditioned repaired / passed candidates: `3 / 3`
- model-conditioned dead-air max: `0.6522 -> 0.0000`
- model-conditioned max added-note ratio: `0.9167`
- model-conditioned max repaired interval: `62`
- model-conditioned dead-air timing repair WAV files: `3`
- model-conditioned dead-air timing repair WAV technical validation: `true`
- model-conditioned remaining wide-interval risk: `true`
- model-conditioned dead-air timing repair objective next decision completed: `true`
- model-conditioned wide-interval follow-up required: `true`
- model-conditioned dead-air target supported: `true`
- model-conditioned pitch-contour repair decision completed: `true`
- model-conditioned pitch-contour selected target: `wide_interval_pitch_contour_repair`
- model-conditioned pitch-contour required interval reduction min: `50`
- model-conditioned pitch-contour repair probe required: `true`
- model-conditioned pitch-contour repair probe completed: `true`
- model-conditioned pitch-contour repaired / passed candidates: `3 / 3`
- model-conditioned pitch-contour max interval: `62 -> 11`
- model-conditioned pitch-contour interval reduction: `51`
- model-conditioned pitch-contour repaired dead-air max: `0.0000`
- model-conditioned pitch-contour max pitch changed ratio: `0.7174`
- model-conditioned pitch-contour WAV files: `3`
- model-conditioned pitch-contour WAV technical validation: `true`
- model-conditioned pitch-contour WAV duration range: `18.422s - 18.978s`
- model-conditioned pitch-contour audio review required: `true`
- model-conditioned pitch-contour listening review package ready: `true`
- model-conditioned pitch-contour listening review items: `3`
- model-conditioned pitch-contour validated review input: `false`
- model-conditioned pitch-contour listening review input guard completed: `true`
- model-conditioned pitch-contour preference fill allowed: `false`
- model-conditioned pitch-contour objective-only next decision completed: `true`
- model-conditioned pitch-contour target supported: `true`
- model-conditioned pitch-contour current evidence consolidation ready: `true`
- model-conditioned pitch-contour changed-ratio review required: `true`
- model-conditioned pitch-contour objective path included in current evidence: `true`
- phrase-bank retrieval baseline completed: `true`
- phrase-bank source records / motifs: `56 / 803`
- phrase-bank exported / qualified MIDI candidates: `3 / 3`
- phrase-bank best notes / unique pitches / max simultaneous: `64 / 22 / 1`
- phrase-bank rendered WAV files: `3`
- phrase-bank audio technical validation: `true`
- phrase-bank listening review package ready: `true`
- phrase-bank listening review items: `3`
- phrase-bank preference fill allowed: `false`
- phrase-bank objective keep candidates: `0 / 3`
- phrase-bank repair required candidates: `3 / 3`
- phrase-bank dead-air range: `0.5873 - 0.6032`
- phrase-bank repaired / qualified candidates: `3 / 3`
- phrase-bank repaired dead-air range: `0.1895 - 0.2211`
- phrase-bank min dead-air gain: `0.3768`
- phrase-bank repaired WAV files: `3`
- phrase-bank repaired audio technical validation: `true`
- phrase-bank repaired listening review package ready: `true`
- phrase-bank repaired listening review items: `3`
- phrase-bank repaired preference fill allowed: `false`
- phrase-bank repaired objective supported candidates: `3 / 3`
- phrase-bank CLI MVP package ready: `true`
- phrase-bank CLI MVP repaired MIDI candidates: `3`
- phrase-bank CLI explicit input smoke completed: `true`
- phrase-bank CLI explicit input context bars: `228`
- phrase-bank CLI explicit input WAV files: `3`
- phrase-bank CLI audio technical validation: `true`
- phrase-bank CLI listening review package ready: `true`
- phrase-bank CLI listening review items: `3`
- phrase-bank CLI validated review input: `false`
- phrase-bank CLI listening review input guard completed: `true`
- phrase-bank CLI preference fill allowed: `false`
- phrase-bank CLI technical MIDI-to-solo path ready: `true`
- phrase-bank CLI MVP current evidence consolidation ready: `true`
- phrase-bank CLI technical path included in current evidence: `true`
- README evidence refreshed: `true`
- MVP completion audit completed: `true`
- MVP completion audit model-conditioned pitch-contour objective included: `true`
- MVP completion audit changed-ratio repair objective included: `true`
- MVP completion audit outside-soloing repair objective included: `true`
- MVP completion audit outside-soloing source context included: `true`
- quality gap decision completed: `true`
- quality gap decision listening review target selected: `true`
- quality gap decision outside-soloing repair objective included: `true`
- quality gap decision outside-soloing source context included: `true`
- pitch-contour changed-ratio review decision completed: `true`
- pitch-contour changed-ratio repair probe required: `true`
- pitch-contour changed-ratio repair probe completed: `true`
- pitch-contour changed-ratio repair passed: `true`
- pitch-contour changed-ratio max: `0.7174 -> 0.4348`
- pitch-contour changed-ratio repaired / passed candidates: `3 / 3`
- pitch-contour changed-ratio repair WAV files: `3`
- pitch-contour changed-ratio repair WAV technical validation: `true`
- pitch-contour changed-ratio repair WAV duration range: `18.422s - 18.978s`
- pitch-contour changed-ratio repair listening review package ready: `true`
- pitch-contour changed-ratio repair listening review items: `3`
- pitch-contour changed-ratio repair validated review input: `false`
- pitch-contour changed-ratio repair listening review input guard completed: `true`
- pitch-contour changed-ratio repair preference fill allowed: `false`
- pitch-contour changed-ratio repair objective-only next decision completed: `true`
- pitch-contour changed-ratio repair objective path supported: `true`
- pitch-contour changed-ratio repair current evidence consolidation ready: `true`
- pitch-contour changed-ratio repair max interval / target: `12 / 12`
- pitch-contour changed-ratio repair max pitch changed ratio / target: `0.4348 / 0.5000`
- current evidence changed-ratio repair objective path included: `true`
- outside-soloing repair objective path included in current evidence: `true`
- outside-soloing repair WAV files: `6`
- outside-soloing repair changed note total: `2`
- outside-soloing source pitch-role risk: `5 -> 2`
- outside-soloing source pitch-role risk delta: `3`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing pitch-role risk count after: `0`
- outside-soloing pitch-role risk delta: `2`
- post-MVP quality iteration plan completed: `true`
- post-MVP selected target: `quality_rubric_baseline`
- post-MVP quality rubric required: `true`
- quality rubric baseline completed: `true`
- quality rubric item / metric group count: `8 / 30`
- quality rubric candidate failure labeling ready: `true`
- candidate failure labeling completed: `true`
- candidate failure failed candidates: `6 / 6`
- candidate failure label type count: `4`
- candidate outside-soloing not evaluable count: `6`
- targeted quality repair sweep completed: `true`
- targeted quality repair failure label count: `12 -> 8`
- targeted quality repair improved candidates: `4 / 6`
- targeted quality repair technical regression count: `0`
- targeted quality repair audio WAV files: `6`
- targeted quality repair audio WAV technical validation: `true`
- targeted quality repair audio duration range: `18.422s - 18.984s`
- outside-soloing repair objective path supported: `true`
- current evidence outside-soloing repair objective path included: `true`
- listening review quality gap completed: `true`
- listening review quality gap open: `true`
- listening review quality gap outside-soloing repair objective included: `true`
- listening review quality gap outside-soloing source context included: `true`
- MVP delivery package completed: `true`
- runnable CLI ready: `true`
- input to ranked MIDI ready: `true`
- input to rendered WAV evidence ready: `true`
- changed-ratio repair audio evidence ready: `true`
- outside-soloing repair evidence ready: `true`
- MVP delivery CLI candidates: `3`
- MVP delivery changed-ratio repair WAV files: `3`
- MVP delivery outside-soloing repair WAV files: `6`
- README final evidence source-context reflected: `true`
- MVP delivery raw artifact upload required: `false`
- model-conditioned input path quality alignment decision completed: `true`
- next boundary: `stage_b_midi_to_solo_readme_final_evidence_refresh`
- validated review input: `false`
- input MIDI -> context -> ranked MIDI -> WAV technical path: `true`
- selected-scale objective repair path complete: `true`
- musical quality MVP completed: `false`
- product MVP completed: `false`
- human/audio preference claim: `false`
- MIDI-to-solo musical quality claim: `false`
- broad trained-model quality claim: `false`
- Brad style adaptation claim: `false`

현재 README는 아래 범위까지만 주장한다.

- 입력 MIDI 기반 context row 생성 가능
- ranked MIDI candidate export 가능
- technical WAV render 가능
- objective MIDI gate 기반 실패/개선 분리 가능
- selected-scale checkpoint repair path의 objective evidence 정리 가능
- model-conditioned strict MIDI/WAV technical evidence 존재
- model-conditioned ranked MIDI candidate export 가능
- model-conditioned ranked WAV technical render 가능
- 입력 MIDI context 기반 phrase-bank retrieval 후보 export 가능
- phrase-bank 후보의 WAV technical render 가능
- phrase-bank 후보의 listening review package 생성 가능
- phrase-bank 후보의 objective-only 실패 신호 분리 가능
- phrase-bank 후보의 dead-air/density objective repair 가능
- phrase-bank repaired MIDI 후보의 WAV technical render 가능
- phrase-bank repaired 후보의 listening review package 생성 가능
- 입력 MIDI 기반 phrase-bank CLI package와 repaired MIDI 후보 export 가능
- 명시적 `--input_midi` 경로 기준 phrase-bank CLI smoke 가능
- 명시적 `--input_midi` 경로 기준 phrase-bank CLI WAV technical render 가능
- 명시적 `--input_midi` 경로 기준 phrase-bank CLI listening review package 생성 가능
- 명시적 `--input_midi` 경로 기준 phrase-bank CLI review input pending 상태에서 preference fill 차단 가능
- 명시적 `--input_midi` 경로 기준 phrase-bank CLI technical path objective decision 가능
- selected-scale objective path와 phrase-bank CLI technical path를 current evidence로 통합 가능
- model-conditioned pitch-contour objective path를 current evidence로 통합 가능
- README 첫 상태 영역에 current evidence와 claim boundary 반영 완료
- technical model-core MVP 완료 범위 audit 가능
- model-conditioned input path alignment decision 가능
- model-conditioned input path probe 가능
- model-conditioned ranked input-path candidate export 가능
- model-conditioned ranked MIDI WAV render 가능
- model-conditioned ranked MIDI/WAV technical replacement consolidation 가능
- model-conditioned ranked MIDI/WAV listening review package 생성 가능
- model-conditioned ranked MIDI 후보의 dead-air/timing repair probe 가능
- model-conditioned dead-air/timing repaired MIDI 후보의 WAV technical render 가능
- model-conditioned dead-air/timing repaired evidence 기준 pitch-contour repair target 정의 가능
- model-conditioned dead-air/timing repaired MIDI 후보의 pitch-contour objective repair 가능
- model-conditioned pitch-contour repaired MIDI 후보의 WAV technical render 가능
- model-conditioned pitch-contour repaired WAV/MIDI 후보의 listening review package 생성 가능
- model-conditioned pitch-contour review input pending 상태에서 preference fill 차단 가능
- model-conditioned pitch-contour objective evidence 기준 current evidence consolidation 경계 결정 가능
- model-conditioned dead-air/timing repaired MIDI/WAV objective next decision 가능
- model-conditioned pitch-contour changed-ratio repair probe 가능
- model-conditioned pitch-contour changed-ratio repaired MIDI 후보의 WAV technical render 가능
- model-conditioned pitch-contour changed-ratio repaired WAV/MIDI 후보의 listening review package 생성 가능
- model-conditioned pitch-contour changed-ratio review input pending 상태에서 preference fill 차단 가능
- model-conditioned pitch-contour changed-ratio objective evidence 기준 current evidence consolidation 경계 결정 가능
- model-conditioned pitch-contour changed-ratio objective path를 current evidence로 통합 가능
- outside-soloing repair objective path를 current evidence로 통합 가능
- listening review quality gap을 delivery package 경계로 분리 가능
- runnable CLI command와 local MIDI/WAV evidence path를 delivery manifest로 기록 가능
- raw MIDI/WAV artifact upload 없이 local output path 기준 MVP delivery package 기록 가능

현재 README가 주장하지 않는 것.

- 최종 jazz solo 품질
- 사용자 청음 선호
- phrase-bank 후보의 청음 품질
- broad training 완료 모델 품질
- Brad Mehldau style adaptation
- realtime DAW/plugin 또는 product-ready improviser

## 현재 evidence

MVP completion audit.

- technical model-core MVP completed: `true`
- input to ranked MIDI completed: `true`
- input to rendered WAV completed: `true`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk count after: `0`
- selected-scale objective repair completed: `true`
- phrase-bank CLI technical path included: `true`
- phrase-bank CLI technical path completed: `true`
- musical quality MVP completed: `false`
- human/audio preference completed: `false`
- product MVP completed: `false`
- next boundary: `stage_b_midi_to_solo_quality_gap_decision`

MVP current evidence refresh.

- boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- next boundary: `stage_b_midi_to_solo_readme_evidence_refresh`
- current MVP evidence supported: `true`
- selected-scale objective path complete: `true`
- phrase-bank CLI technical path ready: `true`
- model-conditioned pitch-contour objective path ready: `true`
- model-conditioned pitch-contour changed-ratio repair objective path ready: `true`
- outside-soloing repair objective path ready: `true`
- outside-soloing repair rendered audio file count: `6`
- outside-soloing repair changed note total: `2`
- outside-soloing source objective pitch-role risk count: `5`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source pitch-role risk delta: `3`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing repair pitch-role risk count after: `0`
- outside-soloing repair pitch-role risk delta: `2`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Quality gap decision.

- boundary: `stage_b_midi_to_solo_quality_gap_decision`
- next boundary: `stage_b_midi_to_solo_listening_review_quality_gap`
- selected target: `listening_review_quality_gap`
- fallback path active: `true`
- model-conditioned input path alignment required: `false`
- phrase-bank CLI technical path completed: `true`
- model-conditioned pitch-contour objective completed: `true`
- model-conditioned pitch-contour changed-ratio repair objective completed: `true`
- model-conditioned pitch-contour max interval / threshold: `11 / 12`
- model-conditioned pitch-contour changed-ratio review required: `true`
- model-conditioned pitch-contour changed-ratio repair max ratio / target: `0.4348 / 0.5000`
- model-conditioned pitch-contour changed-ratio repair max interval / target: `12 / 12`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk count after: `0`
- musical quality MVP completed: `false`
- CLI candidate / rendered WAV: `3 / 3`
- CLI preference fill allowed: `false`
- human review required now: `false`

Listening review quality gap.

- boundary: `stage_b_midi_to_solo_listening_review_quality_gap`
- next boundary: `stage_b_midi_to_solo_mvp_delivery_package`
- selected target: `mvp_delivery_package`
- technical model-core MVP completed: `true`
- changed-ratio repair objective completed: `true`
- changed-ratio repair max ratio / target: `0.4348 / 0.5000`
- changed-ratio repair max interval / target: `12 / 12`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk count after: `0`
- listening review quality gap open: `true`
- technical MVP delivery package ready: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

MVP delivery package.

- boundary: `stage_b_midi_to_solo_mvp_delivery_package`
- next boundary: `stage_b_midi_to_solo_readme_final_evidence_refresh`
- runnable CLI ready: `true`
- input to ranked MIDI ready: `true`
- input to rendered WAV evidence ready: `true`
- changed-ratio repair audio evidence ready: `true`
- outside-soloing repair evidence ready: `true`
- CLI candidate count: `3`
- changed-ratio repair WAV count: `3`
- outside-soloing repair WAV count: `6`
- outside-soloing repair changed note total: `2`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing pitch-role risk count after: `0`
- outside-soloing pitch-role risk delta: `2`
- CLI dead-air ratio range: `0.1895 - 0.2211`
- changed-ratio repair max ratio / target: `0.4348 / 0.5000`
- changed-ratio repair max interval / target: `12 / 12`
- changed-ratio repair WAV duration range: `18.422s - 18.978s`
- raw artifact upload required: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio review decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision`
- next boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe`
- selected target: `lower_pitch_change_ratio_repair_probe`
- repair probe required: `true`
- technical model-core MVP completed: `true`
- model-conditioned pitch-contour objective completed: `true`
- model-conditioned input path alignment required: `false`
- max interval / threshold: `11 / 12`
- changed-ratio review threshold: `0.5`
- changed-ratio review required: `true`
- audio review required: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio repair probe.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package`
- repair strategy: `minimum_change_pitch_class_dynamic_programming`
- repaired / passed candidates: `3 / 3`
- source max pitch changed ratio: `0.7174`
- repaired max pitch changed ratio: `0.4348`
- pitch changed ratio reduction: `0.2826`
- repaired max interval / target: `12 / 12`
- repaired dead-air max: `0.0000`
- min repaired unique pitch count: `24`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio repair audio package.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package`
- rendered audio file count: `3`
- technical WAV validation: `true`
- duration range: `18.422s - 18.978s`
- max repaired pitch changed ratio / target: `0.4348 / 0.5000`
- max repaired interval: `12`
- repaired dead-air max: `0.0000`
- min repaired unique pitch count: `24`
- audio review required: `true`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio repair listening review package.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard`
- package ready: `true`
- review item count: `3`
- validated review input: `false`
- technical WAV validation: `true`
- max repaired pitch changed ratio / target: `0.4348 / 0.5000`
- max repaired interval: `12`
- required input fields: `candidate_rank`, `listening_status`, `preference`, `issue_notes`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio repair listening review input guard.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard`
- source boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision`
- validated review input present: `false`
- preference fill allowed: `false`
- review item count: `3`
- required input field count: `4`
- technical WAV validation: `true`
- max repaired pitch changed ratio / target: `0.4348 / 0.5000`
- max repaired interval: `12`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Pitch-contour changed-ratio repair objective-only next decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision`
- source boundary: `stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard`
- next boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- objective next completed: `true`
- changed-ratio repair objective path supported: `true`
- current evidence consolidation ready: `true`
- technical WAV validation: `true`
- rendered audio file count: `3`
- max repaired pitch changed ratio / target: `0.4348 / 0.5000`
- max repaired interval / target: `12 / 12`
- preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path alignment.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_probe`
- selected probe target: `replace_fallback_with_model_conditioned_input_path_probe`
- model-conditioned input path aligned: `false`
- fallback replacement probe required: `true`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- human review required now: `false`

Model-conditioned input path probe.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_candidate_export`
- model-conditioned source: `model_checkpoint_direct_constrained`
- model-conditioned candidate source available: `true`
- model-conditioned audio technical path available: `true`
- same input context as fallback: `true`
- ranked input-path export contract matched: `false`
- fallback replacement ready: `false`
- candidate export required: `true`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`

Model-conditioned input path candidate export.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_candidate_export`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package`
- generation source: `model_checkpoint_direct_constrained`
- ranked MIDI candidates exported: `true`
- ranked input-path export contract matched: `true`
- exported candidate count: `3`
- best note / unique pitch / max simultaneous: `24 / 20 / 1`
- fallback replacement candidate export ready: `true`
- fallback replacement ready: `false`
- candidate audio render required: `true`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`

Model-conditioned input path audio render package.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation`
- rendered audio file count: `3`
- technical WAV validation: `true`
- model-conditioned ranked audio render completed: `true`
- fallback replacement technical path ready: `true`
- fallback replacement ready: `true`
- WAV duration range: `19.585s - 22.390s`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`

Model-conditioned input path replacement consolidation.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package`
- model-conditioned input to ranked MIDI completed: `true`
- model-conditioned input to ranked WAV completed: `true`
- fallback replacement technical path ready: `true`
- fallback replacement ready: `true`
- listening review package required: `true`
- exported/rendered count: `3 / 3`
- WAV duration range: `19.585s - 22.390s`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path listening review package.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard`
- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_sample_01.wav`, `rank_02_sample_02.wav`, `rank_03_sample_03.wav`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- human/audio preference claimed: `false`

Model-conditioned input path listening review input guard.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard`
- source boundary: `stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision`
- review item count: `3`
- required input field count: `4`
- validated review input present: `false`
- preference fill allowed: `false`
- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path objective-only next decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision`
- model-conditioned technical path ready: `true`
- candidate / exported / rendered: `3 / 3 / 3`
- technical WAV validation: `true`
- dead-air threshold: `0.5000`
- dead-air failure count: `3`
- dead-air min / max: `0.6522 / 0.6522`
- dead-air timing repair required: `true`
- current evidence consolidation ready: `false`
- preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision`
- source boundary: `stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe`
- selected target: `dead_air_timing_continuity`
- repair probe required: `true`
- source dead-air failure count: `3`
- source dead-air min / max: `0.6522 / 0.6522`
- target dead-air max: `0.3500`
- required dead-air gain min: `0.3022`
- strategy: `timing_gap_fill_and_duration_compaction`
- max postprocess removal ratio: `0.2500`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair probe.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package`
- repaired / passed candidates: `3 / 3`
- source dead-air max: `0.6522`
- repaired dead-air max: `0.0000`
- dead-air gain max: `0.6522`
- target dead-air max: `0.3500`
- max added-note ratio: `0.9167`
- max postprocess removal ratio: `0.0000`
- max repaired simultaneous notes: `1`
- max repaired interval: `62`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair audio package.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision`
- rendered audio file count: `3`
- technical WAV validation: `true`
- repaired dead-air max: `0.0000`
- max added-note ratio: `0.9167`
- max postprocess removal ratio: `0.0000`
- max repaired interval: `62`
- remaining wide-interval risk: `true`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair objective next decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision`
- selected target: `wide_interval_pitch_contour_repair`
- technical WAV validation: `true`
- rendered audio file count: `3`
- repaired dead-air max: `0.0000`
- max added-note ratio: `0.9167`
- added-note ratio review required: `true`
- max repaired interval: `62`
- max interval threshold: `12`
- wide-interval follow-up required: `true`
- current evidence consolidation ready: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe`
- selected target: `wide_interval_pitch_contour_repair`
- technical WAV validation: `true`
- dead-air target supported: `true`
- source repaired dead-air max: `0.0000`
- target dead-air max: `0.3500`
- source max added-note ratio: `0.9167`
- added-note ratio review required: `true`
- source max interval: `62`
- target max interval: `12`
- required interval reduction min: `50`
- repair probe required: `true`
- current evidence consolidation ready: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour probe.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package`
- repaired / passed candidates: `3 / 3`
- source max interval: `62`
- repaired max interval: `11`
- target max interval: `12`
- interval reduction: `51`
- required interval reduction min: `50`
- source dead-air max: `0.0000`
- repaired dead-air max: `0.0000`
- min repaired unique pitch count: `22`
- max pitch changed ratio: `0.7174`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour audio package.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package`
- rendered audio file count: `3`
- technical WAV validation: `true`
- duration range: `18.422s - 18.978s`
- repaired dead-air max: `0.0000`
- max repaired interval: `11`
- min repaired unique pitch count: `22`
- max pitch changed ratio: `0.7174`
- audio review required: `true`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour listening review package.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard`
- package ready: `true`
- review item count: `3`
- validated review input: `false`
- technical WAV validation: `true`
- max repaired interval: `11`
- max pitch changed ratio: `0.7174`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour listening review input guard.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard`
- source boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision`
- review item count: `3`
- required input field count: `4`
- validated review input present: `false`
- preference fill allowed: `false`
- technical WAV validation: `true`
- rendered audio file count: `3`
- max repaired interval: `11`
- max pitch changed ratio: `0.7174`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path dead-air timing repair pitch contour objective-only next decision.

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision`
- source boundary: `stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard`
- next boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- review item count: `3`
- validated review input present: `false`
- preference fill allowed: `false`
- technical WAV validation: `true`
- rendered audio file count: `3`
- max repaired interval: `11`
- max interval threshold: `12`
- pitch-contour target supported: `true`
- max pitch changed ratio: `0.7174`
- pitch changed ratio review required: `true`
- audio review required: `true`
- current evidence consolidation ready: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Phrase-bank retrieval baseline.

- generation source: `phrase_bank_data_motif_retrieval`
- source records / motif count: `56 / 803`
- unique rhythm / contour templates: `520 / 328`
- candidate count: `9`
- qualified candidate count: `3`
- exported / exported qualified MIDI candidates: `3 / 3`
- best note / unique pitch / max simultaneous: `64 / 22 / 1`
- best dead-air / phrase coverage: `0.5873015873015873 / 1.0`
- MIDI-to-solo MVP claimed: `false`
- human/audio preference claimed: `false`

Phrase-bank audio render package.

- rendered WAV files: `3`
- technical WAV validation: `true`
- rank 1 duration / sample rate / sha256 prefix: `18.985s / 44100 / 07a95cfe5c4b`
- rank 2 duration / sample rate / sha256 prefix: `18.984s / 44100 / a3a3efc8a9e1`
- rank 3 duration / sample rate / sha256 prefix: `18.997s / 44100 / d3550541fe41`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`

Phrase-bank listening review package.

- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_seed_635.wav`, `rank_02_seed_632.wav`, `rank_03_seed_638.wav`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Phrase-bank listening review input guard.

- validated review input present: `false`
- preference fill allowed: `false`
- review item count: `3`
- required input fields: `candidate_rank`, `listening_status`, `preference`, `issue_notes`
- human/audio preference claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_objective_only_next_decision`

Phrase-bank objective-only next decision.

- review basis: `objective_midi_and_wav_metadata_only`
- candidate count: `3`
- objective keep candidate count: `0`
- repair required candidate count: `3`
- all candidates require repair: `true`
- dead-air range: `0.5873 - 0.6032`
- primary risk flags: `dead_air_ratio_above_review_threshold`, `uniform_bar_note_density`, `low_duration_diversity`, `low_ioi_diversity`, `low_approach_resolution`, `high_pitch_reuse_ratio`, `no_leap_motion`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe`

Phrase-bank dead-air density repair probe.

- repaired candidate count: `3`
- qualified repaired candidate count: `3`
- repair probe target passed: `true`
- repaired dead-air range: `0.1895 - 0.2211`
- dead-air gain range: `0.3768 - 0.3978`
- note count gain: `32`
- per-bar note count pattern: `11 / 13 / 10 / 14 / 11 / 13 / 10 / 14`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package`

Phrase-bank dead-air density repair audio package.

- rendered WAV files: `3`
- technical WAV validation: `true`
- rank 1 duration / sample rate / sha256 prefix: `18.985s / 44100 / 4ac7b2dc9f80`
- rank 2 duration / sample rate / sha256 prefix: `18.984s / 44100 / eb6402477bf3`
- rank 3 duration / sample rate / sha256 prefix: `18.997s / 44100 / 9991eb5b673c`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package`

Phrase-bank dead-air density repair listening review package.

- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_seed_635.wav`, `rank_02_seed_632.wav`, `rank_03_seed_638.wav`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard`

Phrase-bank dead-air density repair listening review input guard.

- validated review input present: `false`
- preference fill allowed: `false`
- review item count: `3`
- required input fields: `candidate_rank`, `listening_status`, `preference`, `issue_notes`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision`

Phrase-bank dead-air density repair objective-only next decision.

- objective supported candidate count: `3`
- all repaired candidates objective supported: `true`
- dead-air range: `0.1895 - 0.2211`
- technical WAV validation: `true`
- CLI MVP package ready: `true`
- preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_mvp_package`

Phrase-bank CLI MVP package.

- CLI MVP package completed: `true`
- ranked repaired MIDI exported: `true`
- candidate count: `3`
- objective supported candidate count: `3`
- dead-air range: `0.1895 - 0.2211`
- input context bars: `8`
- phrase-bank exported candidate count: `3`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke`

Phrase-bank CLI user-input smoke.

- input MIDI: `midi_dataset/midi/studio/Geri Allen/Home Grown/Alone Together.midi`
- explicit input used: `true`
- candidate count: `3`
- objective supported candidate count: `3`
- repaired MIDI file count: `3`
- input context bars: `228`
- dead-air range: `0.1895 - 0.2211`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke`

Phrase-bank CLI audio render smoke.

- render attempted: `true`
- rendered audio file count: `3`
- technical WAV validation: `true`
- CLI user-input audio render completed: `true`
- WAV files: `rank_01_seed_635.wav`, `rank_02_seed_632.wav`, `rank_03_seed_638.wav`
- sample rate: `44100`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_listening_review_package`

Phrase-bank CLI listening review package.

- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_seed_635.wav`, `rank_02_seed_632.wav`, `rank_03_seed_638.wav`
- required input fields: `candidate_rank`, `listening_status`, `preference`, `issue_notes`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard`

Phrase-bank CLI listening review input guard.

- validated review input present: `false`
- preference fill allowed: `false`
- review item count: `3`
- required input field count: `4`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision`

Phrase-bank CLI objective-only next decision.

- technical MIDI-to-solo CLI path ready: `true`
- MVP current evidence consolidation ready: `true`
- explicit input used: `true`
- candidate count: `3`
- objective supported candidate count: `3`
- repaired MIDI file count: `3`
- rendered audio file count: `3`
- technical WAV validation: `true`
- input context bars: `228`
- dead-air range: `0.1895 - 0.2211`
- preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`

MVP current evidence consolidation.

- current MVP evidence supported: `true`
- technical execution evidence supported: `true`
- selected-scale objective path complete: `true`
- phrase-bank CLI technical path ready: `true`
- model-conditioned pitch-contour objective path ready: `true`
- model-conditioned pitch-contour changed-ratio repair objective path ready: `true`
- generation exported / qualified candidates: `3 / 3`
- technical audio rendered WAV files: `3`
- selected-scale objective valid / strict / grammar: `9 / 9 / 9`
- CLI candidate / rendered WAV files: `3 / 3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`
- model-conditioned pitch-contour max interval / threshold: `11 / 12`
- model-conditioned pitch-contour changed-ratio review required: `true`
- model-conditioned pitch-contour changed-ratio repair max ratio / target: `0.4348 / 0.5000`
- model-conditioned pitch-contour changed-ratio repair max interval / target: `12 / 12`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- next boundary: `stage_b_midi_to_solo_readme_evidence_refresh`

README evidence refresh.

- latest evidence boundary reflected: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- technical execution evidence supported: `true`
- selected-scale objective path complete: `true`
- phrase-bank CLI technical path ready: `true`
- model-conditioned pitch-contour objective path ready: `true`
- model-conditioned pitch-contour changed-ratio review required: `true`
- model-conditioned pitch-contour changed-ratio repair objective path ready: `true`
- model-conditioned pitch-contour changed-ratio repair max ratio / target: `0.4348 / 0.5000`
- outside-soloing source-context evidence reflected: `true`
- outside-soloing source pitch-role risk count: `5 -> 2`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk count after: `0`
- quality/preference claim excluded: `true`
- next boundary: `stage_b_midi_to_solo_mvp_completion_audit`

MVP completion audit.

- latest evidence boundary reflected: `stage_b_midi_to_solo_mvp_completion_audit`
- source current evidence boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- technical model-core MVP completed: `true`
- input to ranked MIDI completed: `true`
- input to rendered WAV completed: `true`
- selected-scale objective repair completed: `true`
- phrase-bank CLI technical path completed: `true`
- model-conditioned pitch-contour objective completed: `true`
- model-conditioned pitch-contour max interval / threshold: `11 / 12`
- model-conditioned pitch-contour changed-ratio review required: `true`
- musical quality MVP completed: `false`
- human/audio preference completed: `false`
- product MVP completed: `false`
- next boundary: `stage_b_midi_to_solo_quality_gap_decision`

MIDI-to-solo input contract.

- candidate count: `32`
- exported MIDI candidates: `3`
- target solo bars: `8`
- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`
- fallback path: `phrase_retrieval_data_motif_hybrid`

Input context extraction.

- context bars / events: `8 / 128`
- positions per bar: `16`
- inferred / carry-forward / unknown chord bars: `4 / 4 / 0`
- low-confidence chord bars: `4`
- bass-note bars: `4`

Ranked MIDI generation.

- generation source: `context_conditioned_fallback`
- exported / qualified candidates: `3 / 3`
- best note count: `60`
- best unique pitch count: `14`
- best max simultaneous notes: `1`
- best chord tone ratio: `1.0`

Technical WAV render.

- rendered WAV files: `3`
- sample rate: `44100`
- duration range: `18.617s-18.991s`
- technical WAV validation: `true`

Selected-scale objective repair.

- final boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_path_complete`
- sample / seed count: `9 / 3`
- valid / strict / grammar: `9 / 9 / 9`
- dead-air / collapse failure count: `0 / 0`
- avg / max postprocess removal ratio: `0.21759259259259262 / 0.2916666666666667`
- target avg postprocess removal ratio: `0.3`
- validated review input present: `false`
- preference fill allowed: `false`

## 구현 범위

Dataset and split.

- readable MIDI audit
- candidate file audit
- generic train/val split
- Brad holdout split
- duplicate hash guard

Stage B representation.

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`

Training and checkpoint probe.

- full generic window preparation
- local bounded training smoke
- checkpoint artifact validation
- max sequence budget repair
- broad training claim guard

MIDI-to-solo execution path.

- input MIDI fixture generation
- bar/position/chord/bass context extraction
- checkpoint-conditioned or fallback candidate generation
- constrained monophonic decoding
- objective gate based ranking
- top MIDI export
- local WAV render

Review and repair path.

- note count gate
- unique pitch gate
- max simultaneous note gate
- dead-air ratio gate
- long-note ratio gate
- interval guard
- phrase coverage gate
- collapse warning
- repeatability sweep
- pending listening review guard

## 문제 / 조치 / 관측 결과

`.mid` 존재만으로 성공 판단 위험.

- 관측: one-note collapse, long sustain block, chord block 후보 발생
- 조치: `.mid exists` 성공 조건 제외, objective MIDI review gate 추가
- 결과: note-level failure reason 분리

Stage A representation 한계.

- 관측: `NOTE_ON/OFF` 중심 구조에서 duration/phrase 제어 부족
- 조치: Stage B duration-explicit tokenization 전환
- 결과: `POSITION`, `NOTE_DURATION`, chord context 기반 generation probe 가능

8-bar direct generation sequence budget 부족.

- 관측: 8-bar / 24-note contract tokens `123`, previous max sequence `96`
- 조치: max sequence `160` smoke
- 결과: direct note capacity `17 -> 33`, direct 8-bar strict valid `3/3`

Model-direct candidate phrase failure.

- 관측: max interval `82`, wide interval/register flags `3/3`
- 조치: pitch contour repair
- 결과: max interval `82 -> 9`, wide interval/register flags `0/0`

Model-direct timing/dead-air failure.

- 관측: max dead-air ratio `0.6522`, dead-air flags `3`
- 조치: timing phrase repair
- 결과: max dead-air ratio `0.6522 -> 0.2258`, dead-air flags `3 -> 0`

User listening rejection.

- 관측: preferred rank `3`, overall decision `reject_all`, primary failure `songlike_melody_not_soloing`
- 조치: jazz phrase vocabulary repair target 분리
- 결과: fixed-density / four-note template / duration monotony / IOI monotony / safe interval compression / 4-bar cycle flags `0/0/0/0/0/0`

Selected-scale checkpoint raw generation failure.

- 관측: sample `3`, valid / strict / grammar `0 / 0 / 2`, collapse warning `3`
- 조치: density/grammar/collapse/postprocess repair target 선택
- 결과: valid / strict / grammar `1 / 1 / 3`, note-count/grammar/collapse failure `0 / 0 / 0`

Selected-scale repair repeatability dead-air 병목.

- 관측: seeds `47/52/60`, valid / strict / grammar `2 / 2 / 9`, dead-air failure `7`
- 조치: selected-scale dead-air sustained coverage repair
- 결과: valid / strict / grammar `9 / 9 / 9`, dead-air/collapse `0 / 0`, avg postprocess removal `0.21759259259259262`

청음 claim 과장 위험.

- 관측: rendered WAV와 objective MIDI gate만으로 musical quality 판단 불가
- 조치: validated review input 없을 때 preference fill 차단
- 결과: human/audio preference claim `false`, MIDI-to-solo musical quality claim `false`

## 주요 산출물

Current evidence.

- `docs/STAGE_B_MIDI_TO_SOLO_MVP_CURRENT_EVIDENCE_CONSOLIDATION_2026-06-05.md`
- `outputs/stage_b_midi_to_solo_mvp_current_evidence_consolidation/harness_stage_b_midi_to_solo_mvp_current_evidence_consolidation/stage_b_midi_to_solo_mvp_current_evidence_consolidation.json`

Ranked MIDI candidates.

- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_01_seed_489.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_02_seed_488.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_03_seed_487.mid`

Rendered WAV candidates.

- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_01_seed_489.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_02_seed_488.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_03_seed_487.wav`

## 검증 명령

기본 회귀.

```bash
bash scripts/agent_harness.sh quick
```

현재 MVP evidence consolidation.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-current-evidence-consolidation
```

MIDI-to-solo input contract.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-contract
```

MIDI-to-solo context extraction.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-context-extraction
```

MIDI-to-solo conditioned generation.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-conditioned-generation-probe
```

MIDI-to-solo candidate audio render.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-candidate-audio-render-package
```

Selected-scale objective next decision.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-objective-next
```

## 다음 작업

- Stage B MIDI-to-solo model-conditioned input path probe
- `context_conditioned_fallback` path와 selected-scale objective repair path의 quality gap 분리
- human listening review 입력 전까지 preference/musical quality claim 차단 유지
