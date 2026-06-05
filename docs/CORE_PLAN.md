# Core Plan

작성일: 2026-05-21

이 문서는 이 저장소의 기준 문서다.

흩어진 PR/issue/doc 내용을 하나로 묶어서, 지금 무엇을 만들고 있고 왜 그 순서로 가는지 판단하는 데 사용한다.

## 1. 최종 목표

최종 목표는 symbolic MIDI 기반 jazz piano improvisation model을 만드는 것이다.

장기적으로 만들고 싶은 시스템:

- house/techno/dance groove 위에서 쓸 수 있는 jazz piano solo MIDI generator
- 입력: BPM, chord progression, section, energy, density, optional recent MIDI context
- 출력: 1-2 bar jazz piano solo MIDI
- 사용처: FL Studio, Ableton, piano VST, future live controller
- 방향: generic jazz pianist base를 먼저 만들고, 이후 Brad Mehldau 같은 특정 pianist style adaptation을 검토한다

이 프로젝트는 raw audio generation이 아니다.
지금 단계에서는 DAW plugin, Spring Boot backend, SaaS, UI가 핵심이 아니다.

Long-term reference:

- Live Music Diffusion Models, LMDM, is relevant to the final live-AI-instrument direction.
- The useful ideas are block-wise generation, sliding context, live input/output scheduling, and long-horizon drift control.
- It does not change the current MVP: the present work remains symbolic MIDI jazz solo grammar, reviewability, and phrase quality.

## 2. 현재 MVP 목표

현재 MVP는 제품 MVP가 아니라 model-core MVP다.

MVP 정의:

> 구조적으로 valid하고 리뷰 가능한 1-2 bar jazz piano solo-line MIDI를 생성하는 symbolic MIDI training/generation/evaluation pipeline.

MVP가 끝났다고 볼 수 있는 조건:

- MIDI dataset을 audit하고 train/val split을 관리할 수 있다.
- MIDI를 short phrase/window records로 만들 수 있다.
- tokenized records가 model vocab에 안전하게 들어간다.
- tiny-overfit training이 정상 동작한다.
- generated token을 MIDI로 decode할 수 있다.
- 생성된 MIDI가 단순 파일 생성이 아니라 review gate를 통과한다.
- one-note/two-note output, long sustain block, chord block, empty MIDI를 성공으로 처리하지 않는다.
- 여러 seed/sample에서 pass-rate를 보고 품질을 판단한다.

현재 MVP의 성공 기준은 "멋진 솔로"가 아니다.
먼저 "말이 되는 solo-line 후보"를 안정적으로 만드는 것이다.

2026-05-28 audit 기준:

- pipeline MVP: 완료
- raw trained-model local gate: 완료
- broad trained-model quality: 미검증
- 근거 문서: `docs/STAGE_B_MODEL_CORE_MVP_COMPLETION_AUDIT_2026-05-28.md`
- repair 문서: `docs/STAGE_B_RAW_GENERATION_GATE_REPAIR_2026-05-28.md`
- repeatability 문서: `docs/STAGE_B_RAW_GENERATION_REPEATABILITY_SWEEP_2026-05-28.md`
- dead-air 진단 문서: `docs/STAGE_B_DEAD_AIR_OUTLIER_DIAGNOSTICS_2026-05-28.md`
- candidate gate 문서: `docs/STAGE_B_DEAD_AIR_AWARE_CANDIDATE_GATE_2026-05-28.md`
- broader source 문서: `docs/STAGE_B_BROADER_SOURCE_CANDIDATE_GATE_2026-05-28.md`
- larger source boundary 문서: `docs/STAGE_B_LARGER_SOURCE_RISK_BOUNDARY_2026-05-28.md`
- seed strict margin 진단 문서: `docs/STAGE_B_SEED_STRICT_MARGIN_DIAGNOSTICS_2026-05-28.md`
- seed strict margin warning gate 문서: `docs/STAGE_B_SEED_STRICT_MARGIN_WARNING_GATE_2026-05-28.md`
- candidate count margin recovery 문서: `docs/STAGE_B_CANDIDATE_COUNT_MARGIN_RECOVERY_2026-05-28.md`
- margin-recovered candidate review export 문서: `docs/STAGE_B_MARGIN_RECOVERED_CANDIDATE_REVIEW_EXPORT_2026-05-28.md`
- margin-recovered listening review notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_LISTENING_REVIEW_NOTES_2026-05-28.md`
- margin-recovered proxy review fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PROXY_REVIEW_FILL_2026-05-28.md`
- margin-recovered proxy keep consolidation 문서: `docs/STAGE_B_MARGIN_RECOVERED_PROXY_KEEP_CONSOLIDATION_2026-05-28.md`
- margin-recovered proxy keep focused package 문서: `docs/STAGE_B_MARGIN_RECOVERED_PROXY_KEEP_FOCUSED_PACKAGE_2026-05-28.md`
- margin-recovered focused context decision 문서: `docs/STAGE_B_MARGIN_RECOVERED_FOCUSED_CONTEXT_DECISION_2026-05-28.md`
- margin-recovered focused fallback comparison 문서: `docs/STAGE_B_MARGIN_RECOVERED_FOCUSED_FALLBACK_COMPARISON_2026-05-28.md`
- margin-recovered pitch/dead-air repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PITCH_DEAD_AIR_REPAIR_2026-05-28.md`
- margin-recovered pitch vocabulary sweep 문서: `docs/STAGE_B_MARGIN_RECOVERED_PITCH_VOCAB_SWEEP_2026-05-28.md`
- margin-recovered pitch vocabulary focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_PITCH_VOCAB_FOCUSED_CONTEXT_2026-05-28.md`
- margin-recovered pitch vocabulary focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_PITCH_VOCAB_FOCUSED_LISTENING_NOTES_2026-05-28.md`
- margin-recovered pitch vocabulary focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PITCH_VOCAB_FOCUSED_LISTENING_FILL_2026-05-28.md`
- margin-recovered timing/repetition repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_TIMING_REPETITION_REPAIR_2026-05-28.md`
- margin-recovered timing/repetition focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_TIMING_REPETITION_FOCUSED_CONTEXT_2026-05-28.md`

2026-06-03 MIDI-to-solo execution 기준:

- input MIDI -> context -> ranked MIDI -> WAV technical path: 완료
- current generation source: `model_checkpoint_direct_constrained`
- model-direct sequence budget repair: 완료
- model-direct 8-bar generated MIDI: 생성 완료
- model-direct 8-bar review gate: 통과
- previous scale-smoke checkpoint max_sequence: `96`
- repaired scale-smoke checkpoint max_sequence: `160`
- 8-bar / 24-note minimum contract tokens: `123`
- direct note capacity under previous budget: `17`
- direct note capacity under repaired budget: `33`
- direct 8-bar grammar gate sample count: `3/3`
- direct 8-bar valid sample count: `3/3`
- direct 8-bar strict valid sample count: `3/3`
- min postprocess note count: `24`
- avg postprocess removal ratio: `0.0`
- collapse warning sample rate: `0.0`
- model-direct rendered WAV files: `3`
- model-direct WAV sample rate: `44100`
- model-direct WAV duration range: `19.585s-22.390s`
- model-direct technical WAV validation: `true`
- model-direct MIDI-to-WAV technical path completed: `true`
- model-direct generation quality claimed: `false`
- human/audio preference claimed: `false`
- model-direct phrase diagnostics flags: `dead_air_gap=3`, `wide_interval_contour=3`, `wide_register_span=3`
- model-direct max interval max: `82`
- model-direct max dead-air ratio: `0.6522`
- model-direct pitch contour repair max interval: `82 -> 9`
- model-direct wide interval flag count: `3 -> 0`
- model-direct wide register flag count: `3 -> 0`
- model-direct dead-air flag count: `3 -> 3`
- model-direct timing phrase repair strict valid sample count: `3/3`
- model-direct timing phrase repair dead-air flag count: `3 -> 0`
- model-direct timing phrase repair max dead-air ratio: `0.6522 -> 0.2258`
- model-direct timing phrase repair max interval guard: `9 -> 9`
- model-direct timing phrase repair quality/preference claim: `false`
- model-direct listening review package candidate count: `3`
- model-direct listening review package rendered WAV files: `3`
- model-direct listening review package WAV duration range: `18.926s-19.030s`
- model-direct listening review input template written: `true`
- model-direct listening review completed: `false`
- model-direct human/audio preference claim: `false`
- model-direct user listening review input guard validated input: `false`
- model-direct user listening review input guard preference fill allowed: `false`
- model-direct user listening review input pending fields: status `4`, candidate decision `3`, candidate field `9`
- model-direct user listening review status: `reviewed`
- model-direct user listening review preferred rank: `3`
- model-direct user listening review overall decision: `reject_all`
- model-direct user listening review primary failure: `songlike_melody_not_soloing`
- model-direct human/audio keep claim: `false`
- model-direct MIDI-to-solo musical quality claim: `false`
- model-direct songlike rejection analysis uniform bar density count: `3`
- model-direct songlike rejection analysis four-notes-per-bar template count: `3`
- model-direct songlike rejection analysis duration/IOI monotony count: `3/3`
- model-direct songlike rejection analysis four-bar rhythm cycle repeated count: `3`
- model-direct songlike rejection analysis shared rhythm signature count: `3`
- model-direct songlike rejection analysis max abs interval max: `9`
- model-direct jazz phrase vocabulary repair decision target count: `6`
- model-direct jazz phrase vocabulary repair decision targets: `break_uniform_bar_density`, `replace_shared_rhythm_template`, `reduce_duration_ioi_monotony`, `restore_phrase_vocabulary`, `relax_interval_cap_tradeoff`, `preserve_objective_guards`
- model-direct jazz phrase vocabulary repair decision max allowed interval: `12`
- model-direct jazz phrase vocabulary repair probe target passed: `true`
- model-direct jazz phrase vocabulary repair probe generated MIDI: `3`
- model-direct jazz phrase vocabulary repair probe fixed-density / four-note template / duration monotony / IOI monotony / safe interval compression / 4-bar cycle counts: `0/0/0/0/0/0`
- model-direct jazz phrase vocabulary repair probe shared rhythm signature count: `1`
- model-direct jazz phrase vocabulary repair probe max abs interval max: `12`
- model-direct jazz phrase vocabulary repair probe no overlap: `true`
- model-direct jazz phrase vocabulary repair audio package rendered WAV files: `3`
- model-direct jazz phrase vocabulary repair audio package technical WAV validation: `true`
- model-direct jazz phrase vocabulary repair audio package duration range: `18.975s-18.988s`
- model-direct jazz phrase vocabulary repair audio package listening review completed: `false`
- model-direct jazz phrase vocabulary repair audio package human/audio preference claim: `false`
- model-direct jazz phrase vocabulary repair audio package MIDI-to-solo musical quality claim: `false`
- model-direct jazz phrase vocabulary repair listening review input template written: `true`
- model-direct jazz phrase vocabulary repair listening review validated input: `false`
- model-direct jazz phrase vocabulary repair listening review preference fill allowed: `false`
- model-direct jazz phrase vocabulary repair listening review pending status/candidate decision/candidate field: `4/3/9`
- model-direct jazz phrase vocabulary repair listening review human/audio preference claim: `false`
- model-direct jazz phrase vocabulary repair listening review MIDI-to-solo musical quality claim: `false`
- model-direct jazz phrase vocabulary repair objective-only decision completed: `true`
- model-direct jazz phrase vocabulary repair objective-only stepwise contour bias count: `3`
- model-direct jazz phrase vocabulary repair objective-only distinct density pattern count: `3`
- model-direct jazz phrase vocabulary repair objective-only max abs interval max: `12`
- model-direct jazz phrase vocabulary repair objective-only targets: `reduce_stepwise_contour_bias`, `add_phrase_shape_tension_release`, `add_approach_enclosure_cells`, `preserve_density_variation`, `preserve_interval_guard`, `preserve_no_quality_claim`
- model-direct jazz phrase vocabulary contour phrase-shape repair target passed: `true`
- model-direct jazz phrase vocabulary contour phrase-shape repair stepwise contour bias: `3 -> 0`
- model-direct jazz phrase vocabulary contour phrase-shape repair max small interval ratio <=4: `0.1714`
- model-direct jazz phrase vocabulary contour phrase-shape repair max abs interval max: `11`
- model-direct jazz phrase vocabulary contour phrase-shape repair no overlap: `true`
- model-direct jazz phrase vocabulary contour phrase-shape repair quality claim: `false`
- model-direct jazz phrase vocabulary contour phrase-shape audio package rendered WAV files: `3`
- model-direct jazz phrase vocabulary contour phrase-shape audio package technical WAV validation: `true`
- model-direct jazz phrase vocabulary contour phrase-shape audio package duration range: `18.975s-18.985s`
- model-direct jazz phrase vocabulary contour phrase-shape audio package listening review completed: `false`
- model-direct jazz phrase vocabulary contour phrase-shape audio package human/audio preference claim: `false`
- model-direct jazz phrase vocabulary contour phrase-shape audio package MIDI-to-solo musical quality claim: `false`
- model-direct jazz phrase vocabulary contour phrase-shape listening review input template written: `true`
- model-direct jazz phrase vocabulary contour phrase-shape listening review validated input: `false`
- model-direct jazz phrase vocabulary contour phrase-shape listening review preference fill allowed: `false`
- model-direct jazz phrase vocabulary contour phrase-shape listening review pending status/candidate decision/candidate field: `4/3/9`
- model-direct jazz phrase vocabulary contour phrase-shape listening review human/audio preference claim: `false`
- model-direct jazz phrase vocabulary contour phrase-shape listening review MIDI-to-solo musical quality claim: `false`
- next review target: `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_only_next_decision`
- 근거 문서: `docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_LISTENING_REVIEW_2026-06-04.md`
- margin-recovered timing/repetition focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_TIMING_REPETITION_FOCUSED_LISTENING_NOTES_2026-05-28.md`
- margin-recovered timing/repetition focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_TIMING_REPETITION_FOCUSED_LISTENING_FILL_2026-05-28.md`
- margin-recovered phrase/vocabulary repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_REPAIR_2026-05-28.md`
- margin-recovered phrase/vocabulary focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_CONTEXT_2026-05-28.md`
- margin-recovered phrase/vocabulary focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_LISTENING_NOTES_2026-05-28.md`
- margin-recovered phrase/vocabulary focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_LISTENING_FILL_2026-05-28.md`
- margin-recovered phrase/vocabulary keep consolidation 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_KEEP_CONSOLIDATION_2026-05-28.md`
- margin-recovered phrase/vocabulary keep stability 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_KEEP_STABILITY_2026-05-28.md`
- margin-recovered phrase/vocabulary qualified peer focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_QUALIFIED_PEER_FOCUSED_CONTEXT_2026-05-28.md`
- margin-recovered phrase/vocabulary qualified peer focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_QUALIFIED_PEER_FOCUSED_LISTENING_NOTES_2026-05-28.md`
- margin-recovered phrase/vocabulary qualified peer focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_QUALIFIED_PEER_FOCUSED_LISTENING_FILL_2026-05-28.md`
- margin-recovered phrase/vocabulary two-candidate keep consolidation 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_TWO_CANDIDATE_KEEP_CONSOLIDATION_2026-05-29.md`
- margin-recovered phrase/vocabulary human listening comparison boundary 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_HUMAN_LISTENING_COMPARISON_BOUNDARY_2026-05-29.md`
- margin-recovered phrase/vocabulary duplicate source divergence audit 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DUPLICATE_SOURCE_DIVERGENCE_AUDIT_2026-05-29.md`
- margin-recovered phrase/vocabulary sample-seed diversity repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_SAMPLE_SEED_DIVERSITY_REPAIR_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed repair sweep 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_REPAIR_SWEEP_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_FOCUSED_CONTEXT_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_FOCUSED_LISTENING_NOTES_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_FOCUSED_LISTENING_FILL_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed remaining blocker 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_REMAINING_BLOCKER_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed remaining blocker repair sweep 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_REMAINING_BLOCKER_REPAIR_SWEEP_2026-05-29.md`
- margin-recovered phrase/vocabulary distinct sample-seed dead-air adjacent repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DISTINCT_SAMPLE_SEED_DEAD_AIR_ADJACENT_REPAIR_2026-05-29.md`
- margin-recovered phrase/vocabulary coverage-aware adjacent constrained repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_COVERAGE_AWARE_ADJACENT_CONSTRAINED_REPAIR_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill repair 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_REPAIR_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill focused context 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_FOCUSED_CONTEXT_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill focused listening notes 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_FOCUSED_LISTENING_NOTES_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill focused listening fill 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_FOCUSED_LISTENING_FILL_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill keep consolidation 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_KEEP_CONSOLIDATION_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill human/audio boundary 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_HUMAN_AUDIO_BOUNDARY_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill human/audio review input guard 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_HUMAN_AUDIO_REVIEW_INPUT_GUARD_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill audio review package 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_AUDIO_REVIEW_PACKAGE_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_MIDI_EVIDENCE_REVIEW_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill MIDI evidence consolidation 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_MIDI_EVIDENCE_CONSOLIDATION_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill external human/audio boundary 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_EXTERNAL_HUMAN_AUDIO_BOUNDARY_2026-05-29.md`
- margin-recovered phrase/vocabulary duration coverage fill local audio render package 문서: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_LOCAL_AUDIO_RENDER_PACKAGE_2026-05-29.md`
- local audio render tooling setup 문서: `docs/STAGE_B_LOCAL_AUDIO_RENDER_TOOLING_SETUP_2026-05-29.md`
- renderer path decision 문서: `docs/STAGE_B_RENDERER_PATH_DECISION_2026-05-29.md`
- duration coverage fill local audio render attempt 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-29.md`
- duration coverage fill user listening review fill 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_USER_LISTENING_REVIEW_FILL_2026-05-29.md`
- duration coverage fill user listening review consolidation 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_USER_LISTENING_REVIEW_CONSOLIDATION_2026-05-29.md`
- duration coverage fill next decision 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_NEXT_DECISION_2026-05-29.md`
- duration coverage fill broader repeatability sweep 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_BROADER_REPEATABILITY_SWEEP_2026-05-29.md`
- duration coverage fill dead-air gain repeatability repair 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_DEAD_AIR_GAIN_REPEATABILITY_REPAIR_2026-05-29.md`
- duration coverage fill repeatability consolidation 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_REPEATABILITY_CONSOLIDATION_2026-05-29.md`
- duration coverage fill repeatability audio review package 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_REPEATABILITY_AUDIO_REVIEW_PACKAGE_2026-05-29.md`
- duration coverage fill repeatability user listening review fill 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_REPEATABILITY_USER_LISTENING_REVIEW_FILL_2026-05-29.md`
- duration coverage fill outside-soloing repair decision 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_DECISION_2026-05-29.md`
- duration coverage fill outside-soloing repair sweep 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_SWEEP_2026-05-29.md`
- duration coverage fill outside-soloing repair audio review package 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_AUDIO_REVIEW_PACKAGE_2026-05-29.md`
- duration coverage fill outside-soloing repair user listening review guard 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_USER_LISTENING_REVIEW_GUARD_2026-05-29.md`
- duration coverage fill outside-soloing repair objective evidence consolidation 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_OBJECTIVE_EVIDENCE_CONSOLIDATION_2026-05-29.md`
- duration coverage fill outside-soloing repair next decision 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_NEXT_DECISION_2026-05-29.md`
- duration coverage fill outside-soloing repair broader repeatability sweep 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_BROADER_REPEATABILITY_SWEEP_2026-05-29.md`
- duration coverage fill outside-soloing repair repeatability consolidation 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_REPEATABILITY_CONSOLIDATION_2026-05-29.md`
- duration coverage fill outside-soloing repair final decision 문서: `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_FINAL_DECISION_2026-05-29.md`
- model-core evidence README refresh: `README.md`
- model-core portfolio bullet draft 문서: `docs/STAGE_B_MODEL_CORE_PORTFOLIO_BULLET_DRAFT_2026-05-29.md`
- model-core portfolio bullet refresh 문서: `docs/STAGE_B_MODEL_CORE_PORTFOLIO_BULLET_REFRESH_2026-06-01.md`
- Muzig application wording refresh 문서: `docs/MUZIG_APPLICATION_RESUME_WORDING_REFRESH_2026-06-01.md`
- Muzig application final review package 문서: `docs/MUZIG_APPLICATION_FINAL_REVIEW_PACKAGE_2026-06-01.md`
- MIDI-to-solo MVP input contract 문서: `docs/STAGE_B_MIDI_TO_SOLO_MVP_INPUT_CONTRACT_2026-06-03.md`
- Muzig application resume wording 문서: `docs/MUZIG_APPLICATION_RESUME_WORDING_2026-05-29.md`
- generic base readiness audit 문서: `docs/STAGE_B_GENERIC_BASE_READINESS_AUDIT_2026-05-29.md`
- generic base manifest contract 문서: `docs/STAGE_B_GENERIC_BASE_MANIFEST_CONTRACT_2026-05-29.md`
- generic manifest window smoke 문서: `docs/STAGE_B_GENERIC_MANIFEST_WINDOW_SMOKE_2026-05-29.md`
- generic base tiny training smoke 문서: `docs/STAGE_B_GENERIC_BASE_TINY_TRAINING_SMOKE_2026-05-29.md`
- generic tiny checkpoint generation probe 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_GENERATION_PROBE_2026-05-30.md`
- generic tiny checkpoint grammar repair 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_GRAMMAR_REPAIR_2026-05-30.md`
- generic tiny checkpoint repair repeatability 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_REPEATABILITY_2026-05-30.md`
- generic tiny checkpoint repair review package 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_REVIEW_PACKAGE_2026-05-30.md`
- generic tiny checkpoint repair listening notes 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_LISTENING_NOTES_2026-05-30.md`
- generic tiny checkpoint repair listening fill 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_LISTENING_FILL_2026-05-30.md`
- generic tiny checkpoint repair audio render package 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_AUDIO_RENDER_PACKAGE_2026-05-30.md`
- generic tiny checkpoint repair local audio render attempt 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md`
- generic tiny checkpoint repair user listening review 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_USER_LISTENING_REVIEW_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation decision 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_DECISION_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation sweep 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_SWEEP_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation audio render package 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_AUDIO_RENDER_PACKAGE_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation local audio render attempt 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation MIDI note failure review 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_MIDI_NOTE_FAILURE_REVIEW_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard decision 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_DECISION_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sweep 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SWEEP_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard audio render package 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_AUDIO_RENDER_PACKAGE_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard user listening review 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_USER_LISTENING_REVIEW_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REPAIR_DECISION_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REPAIR_SWEEP_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_AUDIO_RENDER_PACKAGE_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_USER_LISTENING_REVIEW_2026-06-01.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REJECTION_ANALYSIS_2026-06-01.md`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review decision 문서: `docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_MODEL_CORE_REVIEW_DECISION_2026-06-01.md`
- generic model-core training data plan 문서: `docs/STAGE_B_GENERIC_MODEL_CORE_TRAINING_DATA_PLAN_2026-06-01.md`
- generic full manifest window preparation 문서: `docs/STAGE_B_GENERIC_FULL_MANIFEST_WINDOW_PREPARATION_2026-06-01.md`
- generic base training scale smoke 문서: `docs/STAGE_B_GENERIC_BASE_TRAINING_SCALE_SMOKE_2026-06-01.md`
- generic base scale checkpoint generation probe 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_GENERATION_PROBE_2026-06-01.md`
- generic base scale checkpoint grammar representation decision 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_GRAMMAR_REPRESENTATION_DECISION_2026-06-01.md`
- generic base scale checkpoint density coverage repair probe 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DENSITY_COVERAGE_REPAIR_PROBE_2026-06-01.md`
- generic base scale checkpoint density coverage remaining blocker decision 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DENSITY_COVERAGE_REMAINING_BLOCKER_DECISION_2026-06-01.md`
- generic base scale checkpoint duration long-note repair probe 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DURATION_LONG_NOTE_REPAIR_PROBE_2026-06-01.md`
- generic base scale checkpoint duration long-note remaining blocker decision 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DURATION_LONG_NOTE_REMAINING_BLOCKER_DECISION_2026-06-01.md`
- generic base scale checkpoint sustained coverage dead-air repair probe 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_SUSTAINED_COVERAGE_DEAD_AIR_REPAIR_PROBE_2026-06-01.md`
- generic base scale checkpoint objective gate consolidation 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_OBJECTIVE_GATE_CONSOLIDATION_2026-06-01.md`
- generic base scale checkpoint objective gate repeatability sweep 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_OBJECTIVE_GATE_REPEATABILITY_SWEEP_2026-06-01.md`
- generic base scale checkpoint repeatability consolidation 문서: `docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_REPEATABILITY_CONSOLIDATION_2026-06-01.md`
- raw generation gate: `stage-b-generation-probe` 통과
- raw generation repeatability gate: 2-file/3-seed sweep 통과, strict `8/9`
- raw generation dead-air outlier diagnostics: seed `31` sample `1`, dead-air `0.857`, collapse warning false
- raw generation candidate selection gate: selected best seed `17` sample `3`, dead-air `0.333`
- broader source candidate gate: 3-file/3-seed sweep 통과, strict `7/9`, dead-air outlier rate `0.222`
- larger source risk boundary: 4/5/6-file hard gate 통과, 6-file seed `17` strict `1/3`
- seed strict margin diagnostics: 6-file seed `17` failure가 sample `1` dead-air와 sample `2` unique-pitch로 분리됨
- seed strict margin warning gate: hard gate 유지, 6-file warning seed `17` summary 기록
- candidate count margin recovery: 6-file 5-sample run에서 strict `12/15`, warning seed 없음
- margin-recovered candidate review export: seed별 best 후보 3개 objective table 추출, selected best seed `23` sample `1`
- margin-recovered listening review notes: 후보 3개 pending notes template 생성, selected best count `1`
- margin-recovered MIDI proxy review fill: rank `2` seed `31` sample `5` proxy keep, rank `1`은 needs_followup
- margin-recovered proxy keep consolidation: dead-air 단일 기준 selected best와 phrase-rich proxy keep 후보의 claim boundary 정리
- margin-recovered proxy keep focused package: rank `2` 후보 1개를 solo/context review package로 격리, focused max simultaneous notes `1`
- margin-recovered focused context decision: rank `2` proxy keep을 `needs_followup`으로 하향, low pitch variety/dead-air blocker 기록
- margin-recovered focused fallback comparison: 후보 3개 전체 focused context 비교, focused keep `0/3`, low pitch variety `3/3`
- margin-recovered pitch/dead-air repair: 기존 seed `31` checkpoint top_k4 12-sample 재선별, sample `8` dead-air `0.294`, focused unique pitch `5`, remaining flag `low_pitch_variety`
- margin-recovered pitch vocabulary sweep: seed `17/31` top_k5 48개 후보 중 qualified `1`, selected unique pitch `6`, dead-air `0.400`
- margin-recovered pitch vocabulary focused context: selected qualified 후보 focused context decision `keep_for_focused_listening`, flags `{}`
- margin-recovered pitch vocabulary focused listening notes: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, risks `dead_air_ratio_at_gate` / `adjacent_pitch_repeats`
- margin-recovered pitch vocabulary focused listening fill: reviewed `1`, decision `needs_followup`, timing `stiff`, chord fit `strong`, vocabulary `thin`
- margin-recovered timing/repetition repair: seed `37/41` top_k7 temp0.86 96개 후보 중 qualified `2`, selected sample `39`, dead-air `0.353`, adjacent repeats `2`
- margin-recovered timing/repetition focused context: selected repair 후보 focused context decision `keep_for_focused_listening`, flags `{}`
- margin-recovered timing/repetition focused listening notes: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, risks `dead_air_ratio_remaining` / `adjacent_pitch_repeats` / `wide_interval_review`
- margin-recovered timing/repetition focused listening fill: reviewed `1`, decision `needs_followup`, timing `acceptable`, phrase continuation `weak`, jazz vocabulary `thin`
- margin-recovered phrase/vocabulary repair: seed `43/61` top_k7 temp0.82 96개 후보 중 qualified `2`, selected sample `43`, adjacent repeats `0`, max interval `7`, dead-air `0.333`
- margin-recovered phrase/vocabulary focused context: selected repair 후보 focused context decision `keep_for_focused_listening`, flags `{}`, final `C5` over `Fm7` chord tone
- margin-recovered phrase/vocabulary focused listening notes: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, risk `sustained_coverage_review`
- margin-recovered phrase/vocabulary focused listening fill: reviewed `1`, decision `keep`, timing `acceptable`, phrase continuation `acceptable`, jazz vocabulary `acceptable`
- margin-recovered phrase/vocabulary keep consolidation: current evidence keep candidate 정리, proven/not proven boundary 분리
- margin-recovered phrase/vocabulary keep stability: qualified `2/96`, qualified source `2`, stability boundary `narrow_two_source_candidate_support`
- margin-recovered phrase/vocabulary qualified peer focused context: peer candidate context decision `keep_for_focused_listening`, flags `{}`
- margin-recovered phrase/vocabulary qualified peer focused listening notes: peer candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, risk `sustained_coverage_review`
- margin-recovered phrase/vocabulary qualified peer focused listening fill: peer decision `keep`, timing `acceptable`, phrase continuation `acceptable`, jazz vocabulary `acceptable`
- margin-recovered phrase/vocabulary two-candidate keep: selected/peer keep `2`, qualified `2/96`, source `2`, boundary `two_candidate_midi_context_keep_support`
- margin-recovered phrase/vocabulary human listening comparison: human status `pending`, note sequence match `true`, boundary `pending_human_review_same_midi_content`
- margin-recovered phrase/vocabulary duplicate source divergence: source seed diff `true`, shared sample seed `85`, output diversity `absent`
- margin-recovered phrase/vocabulary sample-seed diversity repair: qualified sample seed `1`, distinct peer `0`, boundary `single_distinct_sample_seed_keep_support`
- margin-recovered phrase/vocabulary distinct sample-seed repair sweep: blocked seed `85` 제외, distinct qualified `2`, selected sample seed `155`
- margin-recovered phrase/vocabulary coverage-aware adjacent constrained repair: target-qualified `0/48`, adjacent repeat `0`, dead-air `0.5714`
- margin-recovered phrase/vocabulary duration coverage fill repair: qualified `2/4`, fill additions `6`, dead-air `0.5714 -> 0.2941`
- margin-recovered phrase/vocabulary duration coverage fill focused context: decision `keep_for_focused_listening`, flags `{}`, final `F4` over `Fm7` chord tone
- margin-recovered phrase/vocabulary duration coverage fill focused listening fill: reviewed `1`, decision `keep`, review risks `{}`
- margin-recovered phrase/vocabulary duration coverage fill keep consolidation: boundary `single_postprocess_candidate_keep_support`, human/audio proof 미검증
- margin-recovered phrase/vocabulary duration coverage fill human/audio boundary: source vs fill note sequence match `false`, preference claimed `false`, human/audio status `pending`
- margin-recovered phrase/vocabulary duration coverage fill human/audio review input guard: review input absent, fill status `pending_review_input`, preference `pending`
- margin-recovered phrase/vocabulary duration coverage fill audio review package: status `ready_for_external_review_input`, required files `3`, preference claimed `false`
- margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review: preference `duration_coverage_fill_keep`, score delta `+79.731`, human/audio preference claimed `false`
- margin-recovered phrase/vocabulary duration coverage fill MIDI evidence consolidation: boundary `midi_evidence_preference_support`, human/audio proof 미검증
- margin-recovered phrase/vocabulary duration coverage fill external human/audio boundary: external review status `pending_external_review_input`, human/audio preference claimed `false`
- margin-recovered phrase/vocabulary duration coverage fill local audio render package: planned audio outputs `2`, render attempted `false`, audio quality claim `false`
- local audio render tooling setup: renderer `unavailable`, system modification `false`, audio render attempted `false`
- renderer path decision: decision `renderer_path_or_install_approval_required`, critical user input `true`
- duration coverage fill local audio render attempt: rendered WAV files `2`, sample rate `44100`, duration `6.474s`, preference claim `false`
- duration coverage fill user listening review fill: preference `duration_coverage_fill_keep`, human/audio preference claim `true`, broad model quality claim `false`
- duration coverage fill user listening review consolidation: MIDI evidence and single-user listening both support `duration_coverage_fill_keep`, broad model quality claim `false`
- duration coverage fill next decision: next boundary `broader_repeatability_sweep`, critical user input `false`
- duration coverage fill broader repeatability sweep: distinct sample-seed source `2`, qualified source `2`, variants `7/8`, dead-air improved source `1/2`, boundary `qualified_gate_repeatability_with_partial_dead_air_gain`
- duration coverage fill dead-air gain repeatability repair: selection rule `qualified_dead_air_gain_then_min_fill_additions`, dead-air gain source `2/2`, dead-air gain variants `6/8`, boundary `qualified_gate_repeatability_with_dead_air_gain`
- duration coverage fill repeatability consolidation: current keep single-user preference `true`, distinct source MIDI/dead-air gain support `true`, boundary `current_keep_and_distinct_source_dead_air_gain_midi_support`
- duration coverage fill repeatability audio review package: repeatability source WAV `2`, sample rate `44100`, status `ready_for_user_listening_review`, quality/preference claim `false`
- duration coverage fill repeatability user listening review fill: overall decision `reject_all`, candidate decision `needs_followup`, timing/phrase/vocabulary `outside_or_unclear`, boundary `repeatability_audio_review_needs_followup`
- duration coverage fill outside-soloing repair decision: next boundary `outside_soloing_pitch_role_phrase_clarity_repair`, repair targets `5`, critical user input `false`
- duration coverage fill outside-soloing repair sweep: repaired source `2/2`, qualified variants `6/6`, selected chord-tone ratio `1.000`, max non-chord run `0`, boundary `outside_soloing_pitch_role_repair_candidates`
- duration coverage fill outside-soloing repair audio review package: repaired candidate WAV `2`, sample rate `44100`, status `ready_for_user_listening_review`, quality/preference claim `false`
- duration coverage fill outside-soloing repair user listening review guard: review input `false`, preference claim `false`, objective auto progress `true`, boundary `outside_soloing_repair_audio_review_pending`
- duration coverage fill outside-soloing repair objective evidence consolidation: objective support source `2/2`, chord-tone pass `2/2`, non-chord run pass `2/2`, interval pass `2/2`, preference claim `false`
- duration coverage fill outside-soloing repair next decision: next boundary `outside_soloing_repair_broader_repeatability_sweep`, auto progress `true`, critical user input `false`
- duration coverage fill outside-soloing repair broader repeatability sweep: policy support `3/3`, variants qualified `6/6`, chord-tone min `1.000`, non-chord max `0`, preference claim `false`
- duration coverage fill outside-soloing repair repeatability consolidation: objective source support `2/2`, policy support `3/3`, variants qualified `6/6`, pending review preserved, preference claim `false`
- duration coverage fill outside-soloing repair final decision: final boundary `outside_soloing_repair_objective_path_complete`, next boundary `stage_b_model_core_evidence_readme_refresh`, preference claim `false`
- model-core evidence README refresh: MIDI-to-solo repeatability objective path 기준 README 갱신, generated/qualified `6/6`, flags/overlap `0/0`, quality claim `false`, next boundary `stage_b_midi_to_solo_training_scale_expansion_decision`
- MIDI-to-solo training scale expansion decision: selected train/val `512/128`, max_sequence `160`, objective generated/qualified `6/6`, GPU/cloud spend required `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_training_scale_smoke`
- MIDI-to-solo controlled training scale smoke: train/val `512/128`, max_sequence `160`, best validation loss `5.1061`, checkpoint `1`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe`
- MIDI-to-solo controlled scale checkpoint generation probe: sample `3`, valid/strict/grammar `0/0/3`, collapse warning rate `1.0`, avg/max postprocess removal `0.8090/0.8636`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision`
- MIDI-to-solo controlled scale checkpoint repair decision: selected target `target_density_collapse_postprocess_repair`, postprocess-only/audio/training-scale change selected `false/false/false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe`
- MIDI-to-solo controlled scale checkpoint density/collapse repair probe: note-count/collapse failure `3 -> 0` / `3 -> 0`, avg postprocess removal `0.8090 -> 0.2292`, avg onset/sustained `0.0833/0.1667 -> 0.4583/0.7188`, strict gate `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision`
- MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision: selected target `dead_air_sustained_coverage_repair`, dead-air failure `3/3`, audio/training-scale selected `false/false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe`
- MIDI-to-solo controlled scale checkpoint dead-air repair probe: note groups/bar `12`, valid/strict/grammar `3/3/3`, dead-air failure `3 -> 0`, collapse warning `0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe`
- MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe: seeds `44/52/60`, valid/strict/grammar `7/7/9`, seed `60` partial failure, collapse warning `1`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard decision: selected target `lower_temperature_repeatability_guard_repair`, source/selected temp `0.9 -> 0.75`, top_k `4`, failed seed `[60]`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair probe: temp `0.75`, top_k `4`, seeds `44/52/60`, valid/strict/grammar `9/9/9`, dead-air/collapse `0/0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation: objective support `true`, audio review package required `true`, additional repair `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard audio review package: rendered WAV `3`, duration `6.747s-6.861s`, technical WAV validation `true`, preference claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard listening review: review template `true`, pending status/candidate/field `4/3/9`, preference fill `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_only_next_decision`
- MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard objective-only next decision: objective path support `true`, valid/strict/grammar `9/9/9`, dead-air/collapse `0/0`, preference/quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision`
- MIDI-to-solo controlled scale checkpoint training scale expansion decision: selected train/val `2048/512`, current `512/128`, local bounded smoke, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke`
- MIDI-to-solo controlled scale checkpoint training scale smoke: train/val `2048/512`, best validation loss `3.0396`, checkpoint `1`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe`
- MIDI-to-solo controlled scale checkpoint training scale generation probe: sample `3`, valid/strict/grammar `0/0/2`, collapse warning `3`, avg/max postprocess removal `0.7909/0.8`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision`
- MIDI-to-solo controlled scale checkpoint training scale repair decision: selected target `target_density_grammar_collapse_postprocess_repair`, note-count/collapse/grammar failure `3/3/1`, additional training scale selected `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe`
- MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repair probe: valid/strict/grammar `1/1/3`, note-count/grammar/collapse failure `0/0/0`, avg postprocess removal `0.1875`, target supported `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe`
- MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repeatability probe: seeds `47/52/60`, valid/strict/grammar `2/2/9`, note-count/grammar/collapse failure `0/0/0`, dead-air failure `7`, target support `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision`
- MIDI-to-solo controlled scale checkpoint training scale dead-air remaining blocker decision: selected target `selected_scale_dead_air_sustained_coverage_repair`, dead-air failure `7/9`, density/grammar/collapse follow-up selected `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe`
- MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe: note groups/bar `12`, valid/strict/grammar `3/3/3`, dead-air failure `7 -> 0`, note-count/grammar/collapse failure `0/0/0`, target qualified `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe`
- MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe: seeds `47/52/60`, valid/strict/grammar `7/7/9`, dead-air/collapse failure `2/1`, target qualified `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision`
- MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard decision: selected target `lower_temperature_repeatability_guard_repair`, temp/top_k `0.9/4 -> 0.75/4`, failed seed `[52]`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe`
- MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard repair probe: temp/top_k `0.75/4`, valid/strict/grammar `8/8/9`, dead-air/collapse failure `1/0`, target qualified `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision`
- MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard follow-up decision: selected target `postprocess_removal_dead_air_repair`, valid/strict/grammar `8/8/9`, dead-air/collapse failure `1/0`, avg postprocess removal `0.3611`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe`
- MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair probe: reused-position guard `true`, valid/strict/grammar `9/9/9`, dead-air/collapse failure `0/0`, avg/max postprocess removal `0.2176/0.2917`, target qualified `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation`
- MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation: objective MIDI support `true`, audio review package required `true`, additional repair `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package`
- MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package: candidate/rendered `3/3`, sample rate `44100`, duration `6.866s-6.869s`, technical validation `true`, preference claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review`
- MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair listening review: review template `true`, pending status/candidate/field `4/3/9`, preference fill `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_only_next_decision`
- model-core portfolio bullet refresh: resume bullet `6`, short bullet `3`, generic base checkpoint repeatability `9/9/9`, unsupported claim guard 유지
- Muzig application wording refresh: resume project bullet `5`, short bullet `3`, 자기소개 section `3`, AI 음악 실험/검증 claim만 사용
- Muzig application final review package: long bullet `5`, short bullet `3`, 자기소개 paragraph `3`, 지원 동기 paragraph `2`, 최종 claim check 포함
- MIDI-to-solo MVP input contract: target date `2026-06-11`, candidate count `32`, exported MIDI `3`, target solo bars `8`, fallback `phrase_retrieval_data_motif_hybrid`, next boundary `stage_b_midi_to_solo_context_extraction_mvp`
- MIDI-to-solo context extraction MVP: context bars `8`, context events `128`, inferred/carried/unknown chord bars `4/4/0`, bass-note bars `4`, next boundary `stage_b_midi_to_solo_training_resource_probe`
- MIDI-to-solo training resource probe: ready `true`, context events `128`, full tokenized train/val `154136/21845`, scale-smoke train/val `128/32`, checkpoint count `1`, next boundary `stage_b_midi_to_solo_conditioned_generation_probe`
- MIDI-to-solo conditioned generation probe: source `context_conditioned_fallback`, candidates `8`, exported/qualified `3/3`, best note/unique/max-sim `60/14/1`, next boundary `stage_b_midi_to_solo_candidate_audio_render_package`
- MIDI-to-solo candidate audio render package: rendered WAV `3`, sample rate `44100`, technical validation `true`, preference claim `false`, next boundary `stage_b_midi_to_solo_mvp_execution_consolidation`
- MIDI-to-solo MVP execution consolidation: technical path `true`, source `context_conditioned_fallback`, exported/rendered `3/3`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_generation_repair`
- MIDI-to-solo model-direct monophonic overlap repair: source `model_checkpoint_direct_constrained`, valid/strict `3/3`, avg postprocess removal ratio `0.0`, collapse warning sample rate `0.0`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_audio_render_package`
- MIDI-to-solo model-direct audio render package: rendered WAV `3`, sample rate `44100`, duration range `19.585s-22.390s`, technical validation `true`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_audio_evidence_consolidation`
- MIDI-to-solo model-direct audio evidence consolidation: objective gate `true`, audio render `true`, MIDI-to-WAV technical path `true`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics`
- MIDI-to-solo model-direct phrase quality diagnostics: candidates `3`, flags `dead_air_gap=3`, `wide_interval_contour=3`, `wide_register_span=3`, max interval `82`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair`
- MIDI-to-solo model-direct pitch contour repair: strict `3/3`, max interval `82 -> 9`, wide interval/register flags `3 -> 0`, dead-air flag `3 -> 3`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_timing_phrase_repair`
- MIDI-to-solo model-direct timing phrase repair: strict `3/3`, dead-air flags `3 -> 0`, max dead-air ratio `0.6522 -> 0.2258`, max interval guard `9 -> 9`, quality/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_listening_review_package`
- MIDI-to-solo model-direct listening review package: candidates `3`, rendered WAV `3`, duration range `18.926s-19.030s`, review input template `true`, listening review/preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_user_listening_review_fill`
- MIDI-to-solo model-direct user listening review input guard: validated input `false`, preference fill allowed `false`, pending status/candidate decision/candidate field `4/3/9`, preference claim `false`, next boundary `stage_b_midi_to_solo_model_direct_objective_only_next_decision`
- MIDI-to-solo model-direct user listening review fill: preferred rank `3`, overall `reject_all`, primary failure `songlike_melody_not_soloing`, keep/quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis`
- MIDI-to-solo model-direct songlike melody rejection analysis: uniform density / four-notes template / duration monotony / IOI monotony / interval cap / 4-bar cycle counts `3/3/3/3/3/3`, shared rhythm signature `3`, max interval `9`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision`
- MIDI-to-solo model-direct jazz phrase vocabulary repair decision: target count `6`, distinct rhythm signatures required `true`, max allowed interval `12`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe`
- MIDI-to-solo model-direct jazz phrase vocabulary repair probe: target passed `true`, generated MIDI `3`, fixed-density/four-note/duration/IOI/interval-cap/four-bar-cycle flags `0/0/0/0/0/0`, shared rhythm signature `1`, max interval `12`, no overlap `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package`
- MIDI-to-solo model-direct jazz phrase vocabulary repair audio package: rendered WAV `3`, duration range `18.975s-18.988s`, technical validation `true`, listening review/preference/quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review`
- MIDI-to-solo model-direct jazz phrase vocabulary repair listening review: template `true`, validated input `false`, preference fill `false`, pending fields `4/3/9`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision`
- MIDI-to-solo model-direct jazz phrase vocabulary repair objective-only next decision: stepwise contour bias `3/3`, distinct density pattern `3`, max interval `12`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair: target `true`, stepwise contour bias `3 -> 0`, max interval `11`, no overlap `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape audio package: rendered WAV `3`, duration range `18.975s-18.985s`, technical validation `true`, listening review/preference/quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape listening review: template `true`, validated input `false`, preference fill `false`, pending fields `4/3/9`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_only_next_decision`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-only next decision: current flags `0`, stepwise contour bias `3 -> 0`, additional repair `false`, preference/quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_sweep`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability sweep: sample `6`, qualified `6`, pass rate `1.0000`, current flags `0`, overlap `0`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_consolidation`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability consolidation: support `true`, generated/qualified `6/6`, audio review package required `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability audio package: rendered WAV `6`, duration range `18.865s-19.000s`, technical validation `true`, preference/quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability listening review: template `true`, validated input `false`, preference fill `false`, pending fields `4/6/18`, quality claim `false`, next boundary `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_only_next_decision`
- MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability objective-only next decision: objective path support `true`, generated/qualified `6/6`, flags/overlap `0/0`, pending fields `4/6/18`, preference/quality claim `false`, next boundary `stage_b_model_core_evidence_readme_refresh`
- Stage B model-core evidence README refresh: evidence boundary를 MIDI-to-solo objective path로 갱신, rendered WAV `6`, pending fields `4/6/18`, quality claim `false`, next boundary `stage_b_midi_to_solo_training_scale_expansion_decision`
- Stage B MIDI-to-solo training scale expansion decision: selected train/val `512/128`, prior `128/32`, max_sequence `160`, controlled smoke ready `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_training_scale_smoke`
- Stage B MIDI-to-solo controlled training scale smoke: returncode `0`, best validation loss `5.1061`, checkpoint `1`, vocab fit `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe`
- Stage B MIDI-to-solo controlled scale checkpoint generation probe: generation returncode `0`, sample `3`, valid/strict/grammar `0/0/3`, note count failure `3/3`, collapse warning `3/3`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision`
- Stage B MIDI-to-solo controlled scale checkpoint repair decision: target `density_collapse_postprocess`, all-sample note-count failure `true`, postprocess removal high `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint density/collapse repair probe: target support `true`, note-count failure `0/3`, collapse warning `0/3`, dead-air failure `3/3`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision: target `dead_air_sustained_coverage_repair`, remaining blocker `dead_air_sustained_coverage`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repair probe: target qualified `true`, valid/strict/grammar `3/3/3`, dead-air failure `0/3`, repeatability unverified, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe: target qualified `false`, seed count `3`, strict `7/9`, seed `60` failure, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard decision: target `lower_temperature_repeatability_guard_repair`, strict shortfall `2`, failed seed `[60]`, selected temp/top_k `0.75/4`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair probe: target qualified `true`, strict `9/9`, strict shortfall `2 -> 0`, dead-air/collapse `2/1 -> 0/0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation: objective support `true`, sample `9`, audio review package required `true`, additional repair `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard objective-only next decision: final boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_path_complete`, sample `9`, rendered WAV `3`, pending review `4/3/9`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision`
- Stage B MIDI-to-solo controlled scale checkpoint training scale expansion decision: current `512/128`, selected `2048/512`, max_sequence `160`, full training selected `false`, cloud/GPU spend `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke`
- Stage B MIDI-to-solo controlled scale checkpoint training scale smoke: returncode `0`, best validation loss `3.0396`, checkpoint `1`, vocab fit `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe: generation returncode `0`, sample `3`, valid/strict/grammar `0/0/2`, collapse warning `3/3`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision`
- Stage B MIDI-to-solo controlled scale checkpoint training scale repair decision: target `density_grammar_collapse_postprocess`, note-count/collapse/grammar failure `3/3/1`, additional training scale selected `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repair probe: valid/strict/grammar `1/1/3`, note-count/grammar/collapse failure `0/0/0`, dead-air failure `2`, target support `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repeatability probe: sample `9`, valid/strict/grammar `2/2/9`, dead-air failure `7`, density/grammar/collapse repeatability target support `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air remaining blocker decision: target `selected_scale_dead_air_sustained_coverage_repair`, density/grammar/collapse follow-up `false`, additional scale `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe: target qualified `true`, valid/strict/grammar `3/3/3`, dead-air failure `7 -> 0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe: target qualified `false`, valid/strict/grammar `7/7/9`, dead-air/collapse failure `2/1`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard decision: target `lower_temperature_repeatability_guard_repair`, failed seed `[52]`, selected temp/top_k `0.75/4`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard repair probe: target qualified `false`, valid/strict/grammar `8/8/9`, dead-air/collapse failure `1/0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision`
- Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard follow-up decision: target `postprocess_removal_dead_air_repair`, failed seed `[52]`, avg postprocess removal `0.3611`, target dead-air failure `0`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe`
- Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair probe: target qualified `true`, valid/strict/grammar `9/9/9`, dead-air/collapse failure `0/0`, avg/max postprocess removal `0.2176/0.2917`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation`
- Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation: objective support `true`, audio review package required `true`, additional repair `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package`
- Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package: candidate/rendered `3/3`, sample rate `44100`, duration `6.866s-6.869s`, technical validation `true`, preference claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review`
- Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair listening review: candidate/rendered `3/3`, validated review input `false`, pending fields `4/3/9`, preference fill `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_only_next_decision`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard audio review package: candidate/rendered `3/3`, sample rate `44100`, duration `6.747s-6.861s`, technical validation `true`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review`
- Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard listening review: candidate/rendered `3/3`, validated review input `false`, pending fields `4/3/9`, preference fill `false`, quality claim `false`, next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_only_next_decision`
- Muzig application resume wording: long bullet `7`, short bullet `3`, self-introduction sections `3`, unsupported claim guard 유지
- generic base readiness audit: phase4 prep ready `true`, broad training execution ready `false`, broad quality/Brad adaptation claim `false`
- generic base manifest contract: generic split `2433/270`, Brad split `47/11/14`, leakage/overlap `0`, broad training execution ready `false`
- generic manifest window smoke: selected files `6/3`, tokenized train/val `556/191`, max token id `544 < 547`, broad training execution ready `false`
- generic base tiny training smoke: selected records `32/8`, best validation loss `6.1427`, training returncode `0`, broad quality claim `false`
- generic tiny checkpoint generation probe: command returncode `0`, sample `2`, valid/strict/grammar `0/0/0`, next boundary `grammar repair`
- generic tiny checkpoint grammar repair: baseline valid/strict/grammar `0/0/0`, repair `2/2/2`, constrained quality claim `false`
- generic tiny checkpoint repair repeatability: sample `6`, valid/strict/grammar `5/5/6`, constrained quality claim `false`
- generic tiny checkpoint repair review package: strict-valid candidates `5`, failed rows `1`, musical quality claim `false`
- generic tiny checkpoint repair listening notes: candidate notes `5`, status `pending_human_review`, musical quality claim `false`
- generic tiny checkpoint repair listening fill: review input `false`, fill status `pending_review_input`, candidate `5`, auto progress `true`, musical quality claim `false`
- generic tiny checkpoint repair audio render package: planned audio outputs `5`, render status `ready_for_local_render`, audio quality claim `false`
- generic tiny checkpoint repair local audio render attempt: rendered WAV files `5`, technical WAV validation `true`, audio quality claim `false`
- generic tiny checkpoint repair user listening review: overall `reject_all`, candidate `reject`, primary failure `plunk_and_stop`, keep claim `false`
- generic tiny checkpoint repair phrase continuation decision: repair target `6`, next boundary `phrase_continuation_repair_sweep`, quality claim `false`
- generic tiny checkpoint repair phrase continuation sweep: target qualified `1/6`, selected sample `1` seed `62`, next boundary `phrase_continuation_audio_render_package`, quality claim `false`
- generic tiny checkpoint repair phrase continuation audio render package: planned audio outputs `1`, render status `ready_for_local_render`, audio quality claim `false`
- generic tiny checkpoint repair phrase continuation local audio render attempt: rendered WAV files `1`, technical WAV validation `true`, audio quality claim `false`
- generic tiny checkpoint repair phrase continuation MIDI note failure review: reject_all, pitch span `60`, max interval `60`, large interval ratio `0.875`, next boundary `range_interval_guard_decision`
- generic tiny checkpoint repair phrase continuation range interval guard decision: target pitch span `24`, max interval `12`, large interval ratio `0.35`, severe interval count `0`
- generic tiny checkpoint repair phrase continuation range interval guard sweep: target qualified `3/48`, top cap `9`, sample seed `70`, top span/max interval/large ratio `21/9/0.0`, next boundary `range_interval_guard_audio_render_package`
- generic tiny checkpoint repair phrase continuation range interval guard audio render package: planned outputs `3`, renderer `fluidsynth`, soundfont exists `true`, next boundary `range_interval_guard_local_audio_render_attempt`
- generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt: rendered WAV files `3`, technical validation `true`, duration range `6.818s-7.194s`, next boundary `range_interval_guard_user_listening_review_input`
- generic tiny checkpoint repair phrase continuation range interval guard user listening review: overall `reject_all`, candidate `reject`, primary failure `subjective_not_musical`, next boundary `range_interval_guard_rejection_analysis`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision: primary target `sparse_phrase_continuity_after_range_interval_guard`, next boundary `sparse_phrase_repair_sweep`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep: target qualified candidates `3`, objective gap reduction support `true`, quality claim `false`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package: planned outputs `3`, renderer `fluidsynth`, soundfont exists `true`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt: rendered WAV files `3`, technical validation `true`, duration range `6.792s-7.094s`, next boundary `sparse_phrase_user_listening_review_input`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review: overall `reject_all`, candidate `reject`, primary failure `subjective_not_musical`, keep claim `false`, next boundary `sparse_phrase_rejection_analysis`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis: candidates without objective flags `1/3`, objective proxy gap `true`, next boundary `sparse_phrase_model_core_review_decision`
- generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review decision: continue repair loop `false`, tiny checkpoint `diagnostic_only`, next boundary `generic_model_core_training_data_plan`
- generic model-core training data plan: generic train/val `2433/270`, repair loop `stopped`, next boundary `generic_full_manifest_window_preparation`
- generic full manifest window preparation: tokenized train/val `154136/21845`, max token id/vocab `544/547`, next boundary `generic_base_training_scale_smoke`
- generic base training scale smoke: selected train/val records `128/32`, best validation loss `5.9031`, checkpoint count `1`, next boundary `generic_base_scale_checkpoint_generation_probe`
- generic base scale checkpoint generation probe: sample `3`, valid/strict/grammar `0/0/0`, avg onset/sustained coverage `0.0625/0.09375`, next boundary `generic_base_scale_checkpoint_grammar_representation_decision`
- generic base scale checkpoint grammar representation decision: selected target `target_density_coverage_repair`, note-count failures `3/3`, next boundary `generic_base_scale_checkpoint_density_coverage_repair_probe`
- generic base scale checkpoint density coverage repair probe: repair valid/strict/grammar `1/1/3`, note-count failure delta `3`, coverage delta `0.1042/0.5417`, next boundary `generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision`
- generic base scale checkpoint density coverage remaining blocker decision: selected target `duration_long_note_ratio_repair`, long-note failures `2`, next boundary `generic_base_scale_checkpoint_duration_long_note_repair_probe`
- generic base scale checkpoint duration long-note repair probe: repair valid/strict/grammar `2/2/3`, long-note failure delta `2`, coverage delta `0.0208/-0.2708`, next boundary `generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision`
- generic base scale checkpoint duration long-note remaining blocker decision: selected target `sustained_coverage_dead_air_repair`, dead-air failures `1`, coverage regression `true`, next boundary `generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe`
- generic base scale checkpoint sustained coverage dead-air repair probe: repair valid/strict/grammar `3/3/3`, dead-air failure delta `1`, sustained coverage delta `0.2708`, next boundary `generic_base_scale_checkpoint_objective_gate_consolidation`
- generic base scale checkpoint objective gate consolidation: objective gate support `true`, single seed set only `true`, repeatability claim `false`, next boundary `generic_base_scale_checkpoint_objective_gate_repeatability_sweep`
- generic base scale checkpoint objective gate repeatability sweep: seeds `44/52/60`, valid/strict/grammar `9/9/9`, repeatability claim `true`, quality claim `false`, next boundary `generic_base_scale_checkpoint_repeatability_consolidation`
- generic base scale checkpoint repeatability consolidation: objective MIDI gate repeatability claim `true`, configured seed sweep repeatability claim `true`, quality claim `false`, next boundary `stage_b_model_core_evidence_readme_refresh`
- constrained review gate: `stage-b-overlap-gate` 통과
- focused candidate path: `stage-b-rhythm-phrase-variation` 통과

## 3. 지금까지의 핵심 판단

### 3.1 Stage A는 실패했다

`control_v1` Stage A는 runnable pipeline으로는 검증됐지만, musical output은 실패했다.

관찰된 문제:

- note count가 너무 적음
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line으로 볼 수 없는 구조
- deterministic generation에서 one-note collapse

따라서 Stage A를 더 세게 postprocess하거나 broad training으로 키우지 않는다.

### 3.2 Stage B로 간 이유

Stage B는 REMI/Jazz Transformer 계열 판단을 따른다.

핵심은 모델보다 representation이다.

Stage B에서 명시하는 것:

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`
- tempo/role control

이 방향은 임의로 만든 것이 아니라, REMI, Jazz Transformer, MidiTok 계열의 공통 판단과 맞다.

현재 실패는 Transformer architecture 자체보다 다음 문제에 가깝다.

- NOTE_ON/OFF representation이 duration을 안정적으로 만들지 못함
- full-song sequence가 너무 김
- chord/position/phrase 정보를 모델이 명시적으로 보기 어려움
- 작은 Brad dataset만으로 style을 scratch 학습하기 어려움

### 3.3 지금은 SOTA 재현 단계가 아니다

현재는 Aria, Moonbeam, MidiTok 기반 pretrained model을 붙인 SOTA 구현 단계가 아니다.

지금 하는 일은:

- local tokenizer contract 검증
- phrase/window dataset 검증
- Music Transformer training/generation path 검증
- MIDI decode 검증
- review gate 검증
- collapse/failure mode 측정

즉, 레퍼런스의 원칙을 따른 engineering probe 단계다.

## 4. 현재 상태

현재 main 기준으로 완료된 단계:

1. 전체 jazz piano dataset audit path 작성
2. Brad Mehldau subset audit
3. Stage A `control_v1` training/generation probe
4. Stage A failure review
5. Stage B tokenization spec/test
6. Stage B role dataset preparation
7. Stage B 2-bar phrase/window dataset
8. Stage B vocab/model training path 연결
9. Stage B generation/decode probe
10. Stage B grammar-constrained generation
11. Stage B overlap/dedup postprocess gate
12. Stage B multi-sample review-gate probe
13. Stage B collapse diagnostics and sampling sweep
14. Stage B strict collapse-aware review gate
15. Stage B 2-file Brad generation probe
16. Stage B temporal coverage diagnostics
17. Stage B coverage-aware constrained generation probe
18. Stage B coverage-aware A/B sweep
19. Stage B candidate ranking report
20. Stage B ranking harmonic/repetition gate
21. Stage B chord-aware pitch constrained generation
22. Stage B coverage_chord candidate review export
23. Stage B longer 4-bar coverage_chord phrase probe
24. Stage B phrase contour/repeated-pitch diagnostics
25. Stage B root bias diagnostics
26. Stage B `tones` vs `tones_tensions` pitch-mode comparison
27. Stage B 8-bar approach phrase probe
28. Stage B swing/motif phrase grammar probe
29. Stage B real phrase reference statistics
30. Stage B data-derived motif template extraction
31. Stage B data-derived motif baseline generation
32. Stage B data motif review export
33. Stage B chord-context and straight-grid review export
34. Stage B straight-grid guide-tone/cadence review candidate
35. Stage B data-motif rhythm plus guide-tone/cadence pitch hybrid
36. Stage B reference pitch-role landing statistics and chord-coverage gate
37. Stage B chord progression coverage audit
38. Stage B chord-labeled evaluation subset contract
39. Stage B generated candidate chord-labeled eval bridge
40. Stage B data-guide hybrid generated chord evaluation
41. Stage B review markdown chord eval summary
42. Stage B listening review notes schema
43. Stage B filled listening review aggregate
44. Stage B full review manifest listening notes
45. Stage B objective MIDI note review
46. Stage B objective flags review flow
47. Stage B overlap-free solo-line review export
48. Stage B duration variation review baseline
49. Stage B phrase/cadence review baseline
50. Stage B phrase naturalness objective metrics
51. Stage B phrase recovery review baseline
52. Stage B data motif phrase recovery baseline
53. Stage B objective clean review package
54. Stage B clean context phrase diagnostics
55. Stage B clean listening review notes template
56. Stage B clean MIDI-note proxy review
57. Stage B data-derived contour/cadence landing repair probe
58. Stage B contour repair MIDI-note proxy review
59. Stage B rhythm/phrase vocabulary variation probe
60. Stage B rhythm/phrase variation MIDI-note proxy review
61. Stage B rhythm/phrase variation sample diversity repair
62. Stage B sample-diverse rhythm variation MIDI-note proxy review
63. Stage B rhythm variation timing-grid repetition repair
64. Stage B timing-grid repaired rhythm MIDI-note proxy review
65. Stage B rhythm variation phrase-vocabulary diversity repair
66. Stage B phrase-vocabulary repaired rhythm MIDI-note proxy review
67. Stage B rhythm variation phrase-shape tension repair
68. Stage B phrase-shape tension repaired MIDI-note proxy review
69. Stage B proxy-keep rhythm candidate focused review package
70. Stage B proxy-keep focused context MIDI-note decision
71. Stage B focused context register-arc cadence repair
72. Stage B register-cadence repaired focused proxy review
73. Stage B register-safe phrase vocabulary repair
74. Stage B register-safe phrase vocabulary repaired proxy review
75. Stage B register-safe proxy-keep focused context package
76. Stage B register-safe proxy-keep focused context decision
77. Stage B register-safe focused listening review notes
78. Stage B register-safe focused listening review fill
79. Stage B register-safe timing motif follow-up repair
80. Stage B register-safe timing motif repaired proxy review
81. Stage B data-derived timing phrase vocabulary repair
82. Stage B data-derived timing phrase repaired proxy review
83. Stage B duration/IOI objective repair
84. Stage B duration/IOI repaired proxy review
85. Stage B phrase vocabulary motif variation repair
86. Stage B phrase vocabulary motif variation repaired proxy review
87. Stage B phrase vocabulary motif proxy keep focused package
88. Stage B phrase vocabulary motif focused context decision
89. Stage B phrase vocabulary motif focused listening review notes
90. Stage B phrase vocabulary motif focused listening review fill
91. Stage B focused timing vocabulary follow-up repair
92. Stage B focused timing vocabulary repaired proxy review
93. Stage B focused timing vocabulary proxy keep focused package
94. Stage B focused timing vocabulary focused context decision
95. Stage B focused timing vocabulary focused listening review notes
96. Stage B focused timing vocabulary focused listening review fill
97. Stage B focused timing vocabulary listening follow-up repair
98. Stage B focused timing vocabulary listening follow-up repaired proxy review
99. Stage B focused timing vocabulary follow-up proxy keep focused package
100. Stage B focused timing vocabulary follow-up focused context decision
101. Stage B focused timing vocabulary follow-up focused listening review notes
102. Stage B focused timing vocabulary follow-up focused listening review fill
103. Stage B focused timing vocabulary keep candidate consolidation
104. 포트폴리오용 README 최종 정리
105. README 사무형 문체 정리
106. README 구현 내용 중심 재정리
107. README 하단 참조 섹션 제거
108. Stage B raw generation gate repair
109. Stage B raw generation broader repeatability sweep
110. Stage B raw generation dead-air outlier diagnostics
111. Stage B dead-air-aware candidate selection gate
112. Stage B broader source repeatability with candidate gate
113. Stage B larger source repeatability risk boundary
114. Stage B seed-level strict margin diagnostics
115. Stage B per-seed strict margin warning gate
116. Stage B candidate count margin recovery sweep
117. Stage B margin-recovered candidate review export
118. Stage B margin-recovered candidate listening review notes
119. Stage B margin-recovered MIDI proxy review fill
120. Stage B margin-recovered proxy keep consolidation
121. Stage B margin-recovered proxy keep focused package
122. Stage B margin-recovered focused context decision
123. Stage B margin-recovered focused fallback comparison
124. Stage B margin-recovered pitch/dead-air repair
125. Stage B margin-recovered pitch vocabulary sweep
126. Stage B margin-recovered pitch vocabulary focused context review
127. Stage B margin-recovered pitch vocabulary focused listening notes
128. Stage B margin-recovered pitch vocabulary focused listening fill
129. Stage B margin-recovered pitch vocabulary timing/repetition follow-up repair
130. Stage B margin-recovered timing/repetition focused context review
131. Stage B margin-recovered timing/repetition focused listening notes
132. Stage B margin-recovered timing/repetition focused listening fill
133. Stage B margin-recovered timing/repetition phrase/vocabulary follow-up repair
134. Stage B margin-recovered phrase/vocabulary focused context review
135. Stage B margin-recovered phrase/vocabulary focused listening notes
136. Stage B margin-recovered phrase/vocabulary focused listening fill
137. Stage B margin-recovered phrase/vocabulary keep consolidation
138. Stage B margin-recovered phrase/vocabulary keep stability comparison
139. Stage B margin-recovered phrase/vocabulary qualified peer focused context review
140. Stage B margin-recovered phrase/vocabulary qualified peer focused listening notes
141. Stage B margin-recovered phrase/vocabulary qualified peer focused listening fill
142. Stage B margin-recovered phrase/vocabulary two-candidate keep consolidation
143. Stage B margin-recovered phrase/vocabulary human listening comparison boundary
144. Stage B margin-recovered phrase/vocabulary duplicate-candidate source divergence audit
145. Stage B margin-recovered phrase/vocabulary sample-seed diversity repair
146. Stage B margin-recovered phrase/vocabulary distinct sample-seed repair sweep

가장 최근 의미 있는 결과:

- Issue #43은 candidate MIDI를 직접 읽어 harmonic/repetition diagnostics를 추가했다.
- Issue #43 result: candidates `18`, strict candidates `12`, viable unflagged candidates `0`, flagged candidates `18`
- Issue #45는 constrained generation에서 `NOTE_PITCH` 후보군을 current bar chord 기준으로 제한했다.
- Issue #45 result: candidates `27`, strict candidates `21`, viable unflagged candidates `9`, flagged candidates `18`
- top candidate: `coverage_chord`, groups/bar `4`, sample `2`, score `96.6964`
- top candidate harmonic diagnostics: chord-tone `0.750`, bar chord-tone `0.875`, min bar chord-tone `0.800`, dominant pitch `0.375`, repeated pitch `0.250`
- Issue #47 exported the top 6 `coverage_chord` MIDI candidates to `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe`
- Manual piano-roll review found that these candidates can look like melodic fragments, but are still too short and feel unfinished.
- Issue #49 extends the same coverage+chord-aware setup to a `4` bar probe with `32` note groups per sample and exports direct review candidates from the generation probe report.
- Issue #49 fixes the length problem structurally, but repeated-pitch dependence remains a listening-review risk.
- Issue #51 shows this is not adjacent same-note collapse: adjacent repeated pitch ratio is `0.000`, average direction change ratio is around `0.689`, and max longest same pitch run is `1`.
- Issue #53 shows the perceived "root-heavy" line is not pure root collapse: average root tone ratio is around `0.271`, top candidate root ratio is around `0.219`, but tension ratio is `0.000`.
- Issue #75 shows reference pitch-role stats cannot be trusted yet because known chord note ratio is `0.000`.
- Issue #77 audits the local dataset for chord progression annotations and finds no usable candidate: role meta `2812` scanned with `0` hits, sidecars `0`, MIDI files scanned for text events `120` with `0` chord-text candidates.
- Issue #79 adds a tiny chord-labeled eval contract so known chord labels can produce pitch-role sanity summaries without pretending real Brad/reference labels already exist.
- Issue #81 connects generated candidate reports with known chord metadata to the chord-labeled evaluator.
- Issue #83 applies that bridge to actual data-guide hybrid review candidates and shows `data_motif_guide_tones` has higher chord-tone ratio than `data_motif`.
- Issue #85 writes a combined review markdown so MIDI paths, rhythm metrics, and chord-role metrics can be reviewed together.
- Issue #87 creates a structured listening review notes schema so subjective review can be recorded consistently instead of as loose comments.
- Issue #89 aggregates filled listening review notes into next-step signals and refuses to change generation rules when all candidates are still pending.
- Issue #91 builds listening review notes from the full review manifest so all 15 review candidates, including timing references, have file paths and pending review fields.
- Issue #93 reads generated MIDI notes directly and reports objective flags for overlap/polyphony, grid alignment, scalar/chromatic motion, duration collapse, and chord-role ratios.
- Issue #95 connects objective flags to listening review notes and aggregate priority so problem/warning candidates are visible before manual listening.
- Issue #97 exports overlap-free solo-line review MIDI variants while preserving original sample paths, reducing objective `overlap_polyphonic` from `9` to `0`.
- Issue #99 adds varied-duration review baselines, reducing objective `duration_pattern_collapse` from `6` to `0` while keeping `overlap_polyphonic=0`.
- Issue #101 adds a phrase/cadence review baseline, reducing `chromatic_walk` from `7` to `1` and `too_stepwise_or_scalar` from `6` to `0` in the next review set.
- Issue #103 adds phrase naturalness metrics and reveals that all `12` Issue #101 review candidates have `unresolved_large_leaps`.
- Issue #105 adds a phrase recovery baseline, reducing `phrase_recovery` unresolved large leap ratio to `0.000-0.048`.
- Issue #107 combines data-derived motif rhythm with phrase recovery pitch grammar and keeps `data_motif_phrase_recovery` objective-clean.
- Issue #109 extracts only the objective-clean `data_motif_phrase_recovery` candidates into a focused listening review package.
- Issue #109 result: `3` clean candidates, all with context MIDI paths, note count `63`, unique pitch count `19-23`, unresolved large leap ratio `0.000-0.045`, and tension ratio `0.476-0.524`.
- Issue #111 reads those clean context candidates back at MIDI note-level and reports `3/3` as `listen_with_context`, with no diagnostic flags.
- Issue #113 creates clean listening review notes for those `3` objective-clean context candidates.
- 2026-05-24 MIDI-note proxy review marks the `3` candidates as `needs_followup=2`, `reject=1`, `keep=0`, with contour/landing and rhythm stiffness as the next blockers.
- Issue #115 adds `data_motif_contour_landing_repair`, improving final resolved landing from `1/3` to `3/3` and reducing max interval from `13` to `7` in the comparison harness.
- Issue #115 objective MIDI review reports candidate count `6` and objective flag counts `{}` for the repair-vs-baseline review set.
- Issue #116 contour repair MIDI-note proxy review marks the `6` repair-vs-baseline candidates as `needs_followup=5`, `reject=1`, `keep=0`.
- Issue #116 contour repair aggregate reports `too_stiff=6`, `too_mechanical=6`, `too_repetitive=6`, and recommends phrase vocabulary, timing grid, and motif variation follow-ups.
- Issue #118 adds `data_motif_rhythm_phrase_variation`, improving syncopation `0.625 -> 0.694`, duration diversity `0.062 -> 0.097`, and IOI diversity `0.079 -> 0.115` while keeping objective MIDI flag counts `{}`.
- Issue #118 preserves final landing `3/3`, reduces max interval `7 -> 6`, and keeps unresolved large leap ratio `0.000` for the variation candidates.
- Issue #120 fills MIDI-note proxy review notes for the rhythm/phrase variation candidates and contour repair baseline.
- Issue #120 result: `reviewed=6`, `needs_followup=4`, `reject=2`, `keep=0`, and all candidates still have `timing=too_stiff`.
- Issue #120 finds that the `data_motif_rhythm_phrase_variation` rank 1-3 candidates are exact duplicate note/start/duration sequences, so rank 2 and rank 3 are rejected as duplicate review evidence.
- Issue #122 repairs the variation sample-diversity failure by making seed affect rhythm template choice, slot boundary, duration variation, pitch-cell selection, and approach target.
- Issue #122 review export reports `candidate_count=6`, `unique_note_sequence_count=6`, `duplicate_note_sequence_count=0`, and objective MIDI flag counts `{}`.
- Issue #124 fills MIDI-note proxy review notes for the sample-diverse rhythm variation candidates.
- Issue #124 result: `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`, `too_stiff=6`, and duplicate note sequences remain `0`.
- Issue #126 reduces average most-common IOI ratio from `0.497` to `0.412`, keeps duplicate note sequences at `0`, and keeps objective MIDI flag counts `{}`.
- Issue #126 also removes objective large/unresolved large-leap risk from the variation candidates, but lowers IOI/bar-position/duration diversity.
- Issue #128 fills MIDI-note proxy review notes for the timing-grid repaired candidates and reports `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`, and `too_stiff=6`.
- Issue #128 concludes the next repair should widen phrase vocabulary while preserving duplicate-free/objective-clean timing repair guardrails.
- Issue #130 widens rhythm variation phrase vocabulary while preserving duplicate-free/objective-clean guardrails.
- Issue #130 result: variation `avg_unique_bar_position_pattern_ratio=0.958`, `avg_ioi_diversity_ratio=0.091`, `avg_most_common_ioi_ratio=0.385`, `max_interval=4`, duplicate note sequences `0`, objective flags `{}`.
- Issue #132 fills MIDI-note proxy review notes for the phrase-vocabulary repaired candidates.
- Issue #132 result: `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`, `too_stiff=4`, `acceptable=2`, objective flags `{}`.
- Issue #132 confirms phrase-vocabulary repair should be kept, but next generation work should target phrase shape and tension/approach vocabulary.
- Issue #134 adds phrase target-register shaping and tension pitch-class priority while preserving the Issue #130 rhythm/position guardrails.
- Issue #134 result: variation `avg_tension_ratio=0.437`, `avg_unique_bar_position_pattern_ratio=0.958`, `avg_ioi_diversity_ratio=0.091`, `avg_most_common_ioi_ratio=0.385`, `max_interval=4`, duplicate note sequences `0`, objective flags `{}`.
- Issue #136 fills MIDI-note proxy review notes for the phrase-shape/tension repaired candidates.
- Issue #136 result: `reviewed=6`, `keep=1`, `needs_followup=5`, `reject=0`, objective flags `{}`.
- Issue #136 marks `data_motif_rhythm_phrase_variation_rank_1_sample_3` as the first proxy keep candidate for focused context listening.
- Issue #138 isolates that proxy keep candidate into a focused review package with copied solo/context MIDI and objective first-note summary.
- Issue #138 result: focused package `candidate_count=1`, selected candidate `data_motif_rhythm_phrase_variation_rank_1_sample_3`, objective flags `[]`.
- Issue #140 reviews that single package against context MIDI notes and downgrades it from proxy `keep` to focused context `needs_followup`.
- Issue #140 result: the candidate stays useful as a diagnostic seed, but register arc (`C6` to final `G3`) and cadence/phrase punctuation block a final keep.
- Issue #142 adds focused-context register bounds to `data_motif_rhythm_phrase_variation` so final cadence stays in a right-hand solo register.
- Issue #142 result: variation strict `3/3`, final landing `3/3`, max interval `4`, duplicate note sequences `0`, objective flags `{}`; repaired top candidate ends on `G4` instead of `G3`.
- Issue #144 fills focused proxy review notes for the register-cadence repaired candidates.
- Issue #144 result: `reviewed=6`, `keep=0`, `needs_followup=5`, `reject=1`, objective flags `{}`; repaired top candidate fixes the register blocker but remains boxed-in/cell-like with unique pitch count `18`.
- Issue #146 adds register-safe phrase vocabulary repair to reduce repeated cells without reopening the focused-context register/cadence blocker.
- Issue #146 result: variation strict `3/3`, final landing `3/3`, max interval `4`, duplicate note sequences `0`, objective flags `{}`; top repaired candidate keeps unique pitch count `18` and has `0` exact repeated 4-note cells in the solo review MIDI.
- Issue #148 fills MIDI-note/context proxy review notes for the register-safe phrase vocabulary repaired candidates.
- Issue #148 result: `reviewed=6`, `keep=1`, `needs_followup=4`, `reject=1`, objective flags `{}`; `data_motif_rhythm_phrase_variation_rank_1_sample_3` is restored as a proxy keep candidate for focused context review only.
- Issue #148 aggregate result: `improve_phrase_vocabulary=13`, `fix_timing_grid=8`, `increase_motif_variation=3`, so broad training is still premature.
- Issue #150 isolates that register-safe proxy keep candidate into a focused context review package with copied solo/context MIDI and objective first-note summary.
- Issue #150 result: focused package `candidate_count=1`, selected candidate `data_motif_rhythm_phrase_variation_rank_1_sample_3`, objective flags `[]`, copied MIDI files `2`.
- Issue #152 reviews that single focused package against solo/context MIDI notes and keeps it as `keep_for_focused_listening`.
- Issue #152 result: the prior C6-to-G3 focused-context blocker is gone; remaining risks are repeated pitch-class cells, grid-derived timing, and chromatic color handling that needs real listening review.
- Issue #154 creates a one-candidate focused listening review notes template from the focused package.
- Issue #154 result: candidate count `1`, pending count `1`, proxy decision `keep`; real-listening fields remain pending and must be filled before another generation repair.
- Issue #156 fills that focused review template by MIDI-focused proxy review and downgrades the candidate to `needs_followup`.
- Issue #156 result: timing `stiff`, chord fit `acceptable`, phrase continuation `weak`, landing `acceptable`, jazz vocabulary `thin`; next repair should target timing stiffness, motif variation, and phrase vocabulary while keeping the register-safe final cadence guardrail.
- Issue #158 adds a partial register-safe timing/motif follow-up repair by widening recent phrase memory from `6` to `8` notes and repeated cell penalty lookback from `18` to `32`.
- Issue #158 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective MIDI flags `{}`, avg IOI diversity `0.091`, avg most-common IOI `0.385`, avg tension `0.358`, avg root-tone `0.021`.
- Issue #158 keeps the motif guard but does not claim the timing blocker is solved; asymmetric timing-position changes were excluded because they worsened the metrics.
- Issue #160 fills MIDI-note/context proxy review notes for the Issue #158 repaired candidates.
- Issue #160 result: `reviewed=6`, `keep=0`, `needs_followup=5`, `reject=1`, timing `too_stiff=6`, objective bucket `clean=6`, objective flags `{}`.
- Issue #160 aggregate result: `improve_phrase_vocabulary=16`, `fix_timing_grid=12`, `increase_motif_variation=3`; next generation work should use data-derived timing/phrase vocabulary instead of another local penalty tweak.
- Issue #162 adds data-derived timing row selection for `data_motif_rhythm_phrase_variation` by preferring phrase-like `top_full_templates` while preserving review-safe position/duration shaping.
- Issue #162 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective MIDI flags `{}`, avg syncopation `0.693`, avg tension `0.375`.
- Issue #162 tradeoff: duration diversity fell to `0.073`, IOI diversity fell to `0.079`, and most-common IOI rose to `0.392`; this requires fresh proxy review before promoting the repair.
- Issue #164 fills MIDI-note/context proxy review notes for the Issue #162 repaired candidates.
- Issue #164 result: `reviewed=6`, `keep=0`, `needs_followup=5`, `reject=1`, timing `acceptable=2`, `too_stiff=4`, objective bucket `clean=6`, objective flags `{}`.
- Issue #164 aggregate result: `improve_phrase_vocabulary=16`, `fix_timing_grid=8`, `increase_motif_variation=3`; next generation work should improve duration/IOI objective directly.
- Issue #168 adds phrase-level duration/IOI bar-position planning for `data_motif_rhythm_phrase_variation` and ranks review candidates by IOI diversity before duration diversity.
- Issue #168 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective MIDI flags `{}`, duration diversity `0.078`, IOI diversity `0.111`, tension `0.375`.
- Issue #168 tradeoff: most-common IOI worsened to `0.481`, so this is an objective-diversity repair, not a musical keep.
- Issue #170 fills MIDI-note/context proxy review notes for the Issue #168 repaired candidates.
- Issue #170 result: `reviewed=6`, `keep=0`, `needs_followup=4`, `reject=2`, timing `acceptable=2`, `too_stiff=4`, objective bucket `clean=6`, objective flags `{}`.
- Issue #170 aggregate result: `improve_phrase_vocabulary=12`, `fix_timing_grid=8`, `increase_motif_variation=4`; next generation work should reduce small-cell mechanical contour while preserving objective-clean guardrails.
- Issue #172 repairs phrase vocabulary/motif variation by balancing duration/IOI bar-position patterns and preferring recent pitch reuse avoidance before motif-sized interval preference.
- Issue #172 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective MIDI flags `{}`, unique pitch count `18-20`, duration diversity `0.089`, most-common duration `0.406`, most-common IOI `0.397`.
- Issue #172 tradeoff: IOI diversity falls to `0.095` and source tension ratio falls to `0.318`, so the repaired candidates require fresh proxy review before any keep claim.
- Issue #174 fills MIDI-note/context proxy review notes for the Issue #172 repaired candidates.
- Issue #174 result: `reviewed=6`, `keep=1`, `needs_followup=3`, `reject=2`, timing `acceptable=3`, `too_stiff=3`, objective bucket `clean=6`, objective flags `{}`.
- Issue #174 proxy keep: `data_motif_rhythm_phrase_variation_rank_2_sample_2`, unique pitch count `18`, source most-common IOI `0.397`, objective stepwise ratio `0.460`, objective tension ratio `0.469`, final landing `guide`.
- Issue #174 aggregate result: `improve_phrase_vocabulary=13`, `fix_timing_grid=6`, `increase_motif_variation=3`; next work should isolate the proxy keep candidate into a focused context package before claiming final quality.
- Issue #176 isolates that proxy keep candidate into a focused context review package with copied solo/context MIDI and objective first-note summary.
- Issue #176 result: focused package `candidate_count=1`, selected candidate `data_motif_rhythm_phrase_variation_rank_2_sample_2`, objective flags `[]`, copied MIDI files `2`.
- Issue #178 reviews that focused package against solo/context MIDI notes and keeps it as `keep_for_focused_listening`.
- Issue #178 result: solo range `G3-G5`, final landing `G4`, duplicated 8-note pitch-class chunks `0`, objective flags `[]`; remaining risks are duplicated short pitch-class cells, grid-derived timing, and modest source tension.
- Issue #180 creates a one-candidate focused listening review notes template from the Issue #178 focused-context keep.
- Issue #180 result: focused notes `candidate_count=1`, pending count `1`, proxy decision `keep`; real-listening fields remain pending and must be filled before another generation repair.
- Issue #182 fills that focused listening review note and downgrades the candidate to `needs_followup`.
- Issue #182 result: timing `stiff`, chord fit `acceptable`, phrase continuation `acceptable`, landing `acceptable`, jazz vocabulary `thin`; next repair should target grid-derived timing and short pitch-class vocabulary while preserving focused-context register/cadence guardrails.
- Issue #184 adds a focused timing/vocabulary follow-up repair by blocking replayed 3/4-note pitch-class cells when a safe alternative exists and preserving max interval with repeat fallback.
- Issue #184 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective flags `{}`, unique pitch count `19-20`, stepwise interval ratio `0.460`, root-tone ratio `0.031`.
- Issue #184 tradeoff: rank 1/3 reduce short-cell repetition, but rank 2 introduces more adjacent pitch repeat; this requires fresh proxy review before any keep claim.
- Issue #186 fills MIDI-note/context proxy review notes for the Issue #184 repaired candidates.
- Issue #186 result: `reviewed=6`, `keep=1`, `needs_followup=3`, `reject=2`, timing `acceptable=3`, `too_stiff=3`, objective bucket `clean=6`, objective flags `{}`.
- Issue #186 proxy keep: `data_motif_rhythm_phrase_variation_rank_3_sample_3`, unique pitch count `20`, max interval `4`, final landing `guide`, source most-common IOI `0.397`, objective stepwise ratio `0.460`, objective tension ratio `0.453`.
- Issue #186 aggregate result: `improve_phrase_vocabulary=12`, `fix_timing_grid=6`, `increase_motif_variation=4`; next work should isolate the proxy keep candidate into a focused context package before claiming final quality.
- Issue #188 isolates that proxy keep candidate into a focused context review package with copied solo/context MIDI and objective first-note summary.
- Issue #188 result: focused package `candidate_count=1`, selected candidate `data_motif_rhythm_phrase_variation_rank_3_sample_3`, objective flags `[]`, copied MIDI files `2`.
- Issue #190 reviews that focused package against solo/context MIDI notes and keeps it as `keep_for_focused_listening`.
- Issue #190 result: solo range `G3-G5`, final landing `D5` over `Ebmaj7`, max interval `4`, duplicated 4/8-note pitch-class chunks `0`, objective flags `[]`; remaining risks are adjacent repeats, duplicated 3-note cells, quantized timing, and low source tension.
- Issue #192 creates a one-candidate focused listening review notes template from the Issue #190 focused-context keep.
- Issue #192 result: focused notes `candidate_count=1`, pending count `1`, proxy decision `keep`; real-listening fields remain pending and must be filled before another generation repair.
- Issue #194 fills that focused listening review note and downgrades the candidate to `needs_followup`.
- Issue #194 result: timing `stiff`, chord fit `acceptable`, phrase continuation `acceptable`, landing `strong`, jazz vocabulary `thin`; next repair should target adjacent repeats, duplicated 3-note cells, timing stiffness, and chord-color/tension while preserving focused-context register/cadence guardrails.
- Issue #196 adds a focused listening follow-up repair by avoiding immediate pitch-class reuse when safe alternatives exist and by trying tension/recovery/next-guide alternatives before repeat fallback.
- Issue #196 result: variation strict `3/3`, final landing `3/3`, max interval `4`, objective flags `{}`, adjacent pitch repeats reduced to `0` for all three repaired candidates.
- Issue #196 tradeoff: rank 2 improves duplicated 3/4-note cells to `0`, but avg source tension falls to `0.307` and rank 1/3 duplicated 3-note cells increase; this requires fresh proxy review before any keep claim.
- Issue #198 fills MIDI-note/context proxy review notes for the Issue #196 repaired candidates.
- Issue #198 result: `reviewed=6`, `keep=1`, `needs_followup=3`, `reject=2`, timing `acceptable=3`, `too_stiff=3`, objective bucket `clean=6`, objective flags `{}`.
- Issue #198 proxy keep: `data_motif_rhythm_phrase_variation_rank_2_sample_2`, adjacent repeats `0`, duplicated 3/4/8-note cells `0`, final landing `D5`, max interval `4`, objective tension `0.469`.
- Issue #200 isolates that proxy keep candidate into a focused context review package with copied solo/context MIDI and objective first-note summary.
- Issue #200 result: focused package `candidate_count=1`, selected candidate `data_motif_rhythm_phrase_variation_rank_2_sample_2`, objective flags `[]`, copied MIDI files `2`.
- Issue #204 reviews that focused package against solo/context MIDI notes and keeps it as `keep_for_focused_listening`.
- Issue #204 result: solo range `G3-G5`, final landing `D5` over `Ebmaj7`, max interval `4`, adjacent repeats `0`, duplicated 3/4/8-note pitch-class chunks `0`, objective flags `[]`; remaining risks are mechanical timing, low IOI diversity, and moderate source tension.
- Issue #206 creates a one-candidate focused listening review notes template from the Issue #204 focused-context keep.
- Issue #206 result: focused notes `candidate_count=1`, pending count `1`, proxy decision `keep`, proxy issue `too_mechanical`; real-listening fields remain pending and must be filled before another generation repair.
- Issue #208 fills that focused listening review note and keeps the candidate as the current best focused review candidate.
- Issue #208 result: timing `acceptable`, chord fit `strong`, phrase continuation `acceptable`, landing `strong`, jazz vocabulary `acceptable`, decision `keep`; this remains a single-candidate focused keep, not broad model-quality proof.
- Issue #210 consolidates that keep candidate as the current reviewable MIDI outcome and separates proven evidence from non-proven claims.
- Issue #210 result: current pipeline has a single focused keep candidate with objective-clean status, zero adjacent repeats, zero duplicated 3/4/8-note pitch-class chunks, and focused review `keep`; broad model quality, human/audio preference, multi-seed reliability, and style adaptation remain unproven.
- Issue #212 rewrites README as a portfolio-facing Korean project document.
- Issue #212 result: README now leads with problem definition, validation approach, current focused keep evidence, conservative claim boundaries, execution commands, and portfolio talking points.
- Issue #214 rewrites README into a noun-based business style.
- Issue #214 result: README now uses tables, bullets, and concise noun phrases instead of narrative paragraphs while preserving conservative claim boundaries.
- Issue #216 rewrites README around implemented components and problem-solving evidence.
- Issue #216 result: README now shows what was built, what failed, how it was fixed, and what the measured result is; the previous portfolio-point section is removed.
- Issue #218 removes the README footer reference sections.
- Issue #218 result: README now ends at execution commands and keeps the focus on implementation, problem solving, validation, and results.
- Issue #254 runs a top_k4 12-sample repair from the existing seed `31` checkpoint and selects sample `8` as the best partial pitch/dead-air repair.
- Issue #254 result: dead-air improves from `0.444` to `0.294`, focused unique pitch improves from `4` to `5`, and remaining flag is `low_pitch_variety`; this is not a focused keep.
- Issue #256 runs a seed/top-k pitch vocabulary sweep over `48` candidates and finds `1` qualified candidate.
- Issue #256 result: selected sample has focused unique pitch `6`, dead-air `0.400`, focused notes `13`, duplicated 3-note chunks `0`, but dead-air and adjacent repeats regress from Issue #254.
- Issue #258 isolates that selected pitch-vocabulary candidate into focused solo/context review and marks it `keep_for_focused_listening`.
- Issue #258 result: focused context flags `{}`, max active `1`, final `G#4` over `Fm7` chord tone, with dead-air `0.400` and adjacent repeats `3` kept as listening-review risks.
- Issue #260 creates focused listening review notes for that candidate with pending listening fields and explicit risks.
- Issue #260 result: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, risks `dead_air_ratio_at_gate` and `adjacent_pitch_repeats`.
- Issue #262 fills the focused listening notes from MIDI/context evidence and downgrades the candidate to `needs_followup`.
- Issue #262 result: timing `stiff`, chord fit `strong`, phrase continuation `weak`, landing `strong`, jazz vocabulary `thin`.
- Issue #264 runs a top_k7 temperature `0.86` timing/repetition sweep over seed `37/41`.
- Issue #264 result: selected sample `39` keeps focused unique pitch `7`, max active `1`, duplicated 3-note chunks `0`, and improves dead-air `0.400 -> 0.353` plus adjacent repeats `3 -> 2`; focused context/listening 재검증은 아직 남아 있다.
- Issue #266 isolates that timing/repetition repair candidate into a focused solo/context package and reviews it against context MIDI.
- Issue #266 result: focused context decision `keep_for_focused_listening`, flags `{}`, note count `14`, unique pitch `7`, range `C#4-G5`, phrase span `6.5` beats, final `A#4` over `Fm7` tension.
- Issue #268 creates focused listening notes for that context keep candidate.
- Issue #268 result: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, review risks `dead_air_ratio_remaining`, `adjacent_pitch_repeats`, and `wide_interval_review`.
- Issue #270 fills that focused listening note from MIDI/context evidence.
- Issue #270 result: timing improves to `acceptable`, chord fit `acceptable`, landing `acceptable`, but phrase continuation is `weak`, jazz vocabulary is `thin`, and decision remains `needs_followup` because adjacent repeats `2` and max interval `16` remain.
- Issue #272 runs a phrase/vocabulary repair sweep over seed `43/61`, top_k `7`, temperature `0.82`.
- Issue #272 result: selected sample `43` keeps dead-air `< 0.400`, focused unique pitch `8`, focused notes `13`, max active `1`, dup3 `0`, and improves adjacent repeats `2 -> 0` plus max interval `16 -> 7`; focused context/listening 재검증은 아직 남아 있다.
- Issue #274 isolates that phrase/vocabulary repair candidate into a focused solo/context package and reviews it against context MIDI.
- Issue #274 result: focused context decision `keep_for_focused_listening`, flags `{}`, note count `13`, unique pitch `8`, range `G4-E5`, phrase span `7.0` beats, max active `1`, final `C5` over `Fm7` chord tone.
- Issue #276 creates focused listening notes for that context keep candidate.
- Issue #276 result: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, review risk `sustained_coverage_review`; adjacent repeat and wide interval risks do not reappear.
- Issue #278 fills that focused listening note from MIDI/context evidence.
- Issue #278 result: reviewed `1`, decision `keep`, timing `acceptable`, chord fit `strong`, phrase continuation `acceptable`, landing `strong`, jazz vocabulary `acceptable`; sustained coverage risk remains documented and this is not human/audio proof.
- Issue #280 consolidates that result as the current margin-recovered evidence keep candidate.
- Issue #280 result: candidate `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` is documented as evidence keep, while human/audio preference, broad trained-model quality, Brad style adaptation, and broader repeatability remain unproven.
- Issue #282 compares that keep candidate against the Issue #272 phrase/vocabulary sweep.
- Issue #282 result: qualified `2/96`, qualified source count `2`, selected keep plus peer `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25`, stability boundary `narrow_two_source_candidate_support`.
- Issue #284 isolates that qualified peer into a focused solo/context package and reviews it against context MIDI.
- Issue #284 result: peer context decision `keep_for_focused_listening`, flags `{}`, note count `13`, unique pitch `8`, range `G4-E5`, phrase span `7.0` beats, final `C5` over `Fm7` chord tone.
- Issue #286 creates focused listening notes for that peer context keep candidate.
- Issue #286 result: candidate `1`, pending `1`, prior decision `keep_for_focused_listening`, review risk `sustained_coverage_review`.
- Issue #288 fills that peer focused listening note from MIDI/context evidence.
- Issue #288 result: peer decision `keep`, timing `acceptable`, chord fit `strong`, phrase continuation `acceptable`, landing `strong`, jazz vocabulary `acceptable`; selected and peer candidates now both have filled evidence keep decisions.
- Issue #290 joins the keep stability summary with selected and peer filled notes.
- Issue #290 result: keep candidates `2`, qualified `2/96`, qualified sources `2`, boundary `two_candidate_midi_context_keep_support`; this remains MIDI/context evidence, not human/audio proof.
- Issue #292 prepares the selected/peer pair for human listening comparison and keeps preference fields pending.
- Issue #292 result: note sequence match `true`, metric fingerprint match `true`, boundary `pending_human_review_same_midi_content`; same-render A/B preference is not meaningful until source divergence is audited.
- Issue #294 audits source divergence for the duplicate selected/peer pair.
- Issue #294 result: source seed diff `true`, sample index diff `true`, shared sample seed `85`, note sequence match `true`, boundary `shared_sample_seed_duplicate_output`; this is not two distinct musical outputs.
- Issue #296 repairs the sample-seed diversity claim boundary.
- Issue #296 result: qualified source seed count `2`, qualified sample seed count `1`, distinct peer count `0`, boundary `single_distinct_sample_seed_keep_support`.
- Issue #298 runs a focused checkpoint-based sweep with sample seed ranges outside duplicate seed `85`.
- Issue #298 result: qualified `2/96`, distinct sample-seed qualified `2`, selected sample seed `155`, boundary `distinct_sample_seed_qualified_candidate_found`.
- Issue #300 packages the selected distinct sample-seed candidate into focused solo/context artifacts and runs context decision.
- Issue #300 result: decision `keep_for_focused_listening`, flags `{}`, final `D5` over `Fm7` tension, max active `1`.
- Issue #302 writes focused listening notes for the distinct sample-seed context keep candidate.
- Issue #302 result: candidate `1`, pending `1`, review risks `dead_air_ratio_remaining` and `adjacent_pitch_repeats`.
- Issue #304 fills the distinct sample-seed notes from MIDI/context evidence.
- Issue #304 result: decision `needs_followup`, phrase continuation `weak`, jazz vocabulary `thin`; timing and landing remain acceptable.
- Issue #306 summarizes the remaining blockers into the next repair target.
- Issue #306 result: blockers `phrase_continuation_weak`, `jazz_vocabulary_thin`, `short_phrase_span`, `pitch_variety_floor`, `adjacent_pitch_repeats`; target phrase span `>= 7.0`, unique pitch `>= 7`, adjacent repeats `0`.
- Issue #308 runs an additional checkpoint-based repair sweep against the Issue #306 target.
- Issue #308 result: target-qualified `0/96`; best partial candidate sample seed `250`, focused unique pitch `9`, dead-air `0.3889`, adjacent repeats `1`.
- Issue #310 runs a lower-temperature/top_k targeted repair sweep for dead-air and adjacent repeats.
- Issue #310 result: target-qualified `0/96`; best partial candidate sample seed `341`, focused unique pitch `7`, dead-air `0.3889`, adjacent repeats `1`, max interval `7`.
- Issue #312 runs coverage-aware constrained decoding with chord-aware repeat window.
- Issue #312 result: target-qualified `0/48`; best partial candidate sample seed `355`, focused unique pitch `9`, dead-air `0.5714`, adjacent repeats `0`, max interval `7`.
- 이것은 아직 unconstrained model quality나 Brad style adaptation 성공을 의미하지 않는다.

중요한 해석:

> 지금은 "모델이 된다"가 아니라 "어떤 representation/generation constraint에서 reviewable MIDI가 되는지 측정할 수 있게 됐다"가 성과다.

## 5. 현재 가장 큰 위험

가장 큰 위험은 postprocess와 constrained generation으로 gate만 통과시키고, 실제 모델 품질은 좋아지지 않는 것이다.

현재 결과는 이 위험을 보여준다.

- grammar gate는 통과할 수 있다.
- MIDI 파일도 생성된다.
- overlap postprocess 후 review gate를 일부 통과한다.
- 하지만 `top_k=1`에서는 같은 position/pitch 반복 collapse가 발생한다.

따라서 다음 단계도 곧바로 broad training이 아니다.
Issue #312는 constrained decoding으로 adjacent repeat를 줄였지만 dead-air가 악화되어 target-qualified 후보를 찾지 못했다.
다음 작업은 duration/coverage fill repair로 dead-air를 직접 낮추는 것이다.

## 6. 다음 단계 로드맵

### Phase 1. Collapse Diagnostics

목표:

- 반복 position/pitch collapse를 metric으로 잡는다.
- postprocess 후 살아남은 note 수만 보지 않는다.
- 생성 전 token 수준과 생성 후 MIDI 수준을 모두 분석한다.

구현 후보:

- repeated `POSITION + NOTE_PITCH` pair ratio
- repeated pitch ratio
- unique position count
- unique pitch count
- per-bar note distribution
- postprocess removal ratio
- sample diversity score

통과 기준:

- collapse report가 `report.json`에 들어간다.
- invalid 샘플의 이유가 "note count low"보다 더 구체적으로 나온다.
- `top_k=1`, `top_k=2` 실패 차이를 숫자로 설명할 수 있다.

### Phase 2. Sampling Sweep

목표:

- 한 checkpoint에서 sampling parameter가 품질에 주는 영향을 측정한다.

비교 후보:

- `top_k=1`
- `top_k=2`
- `top_k=4`
- temperature `0.7`, `0.9`, `1.1`

통과 기준:

- 각 설정별 sample count, grammar pass rate, valid pass rate를 비교한다.
- best sample 하나가 아니라 pass-rate table로 판단한다.
- MIDI를 들어볼 후보를 자동으로 고른다.

### Phase 3. Stage B 2-File Brad Probe

목표:

- one-file tiny smoke를 넘어서 Brad 2-file Stage B probe를 실행한다.
- Stage A에서 실패했던 2-file 조건을 Stage B representation으로 다시 비교한다.

통과 기준:

- train/val split이 명확하다.
- 2-file window dataset이 정상 생성된다.
- 여러 seed/sample에서 최소 pass-rate를 만족한다.
- piano roll에서 one-note/chord-block/sustain-block이 아니다.

실패하면:

- postprocess를 더 세게 하지 않는다.
- tokenization 또는 model/data scale 문제로 본다.

### Phase 3.5. Temporal Coverage and Coverage-Aware Generation

목표:

- 2-file Brad probe의 dead-air failure를 token-level temporal coverage로 설명한다.
- sparse onset, tail/head empty span, sustained empty run을 sample report에 기록한다.
- constrained generation의 `POSITION` 선택만 coverage-aware로 바꿔 review gate 통과 가능성을 검증한다.

현재 결과:

- temporal diagnostics: grammar `3/3`, basic `0/3`, strict `0/3`, max longest sustained empty run `11`
- coverage-aware constrained generation: grammar `3/3`, basic `3/3`, strict `3/3`, max longest sustained empty run `6`

다음 통과 기준:

- completed: plain constrained vs coverage-aware constrained A/B sweep을 같은 checkpoint 조건에서 비교했다.
- completed: `note_groups_per_bar=4/6/8`을 비교했다.
- next: pass-rate가 좋아져도 이것을 style learning 성공으로 표현하지 않고 candidate ranking 기준을 추가한다.

### Phase 3.6. Harmonic Candidate Gate and Pitch Control

목표:

- strict gate를 통과했지만 실제 piano roll에서 solo-line이 아닌 후보를 걸러낸다.
- bar-level chord-tone ratio, dominant pitch ratio, repeated pitch ratio, repeated onset-template ratio를 ranking에 반영한다.
- ranking만 보정하지 않고 generation-side pitch/harmony 제어로 넘어갈 기준을 만든다.

현재 결과:

- completed: ranking이 candidate MIDI를 직접 읽어 harmonic/repetition diagnostics를 계산한다.
- completed: low chord-tone/repeated pitch/mechanical template 후보에 review flags를 붙인다.
- latest result: `18` candidates 중 viable unflagged candidate `0`.
- completed: chord-aware pitch constrained generation을 추가했다.
- latest chord-aware result: `27` candidates 중 viable unflagged candidate `9`.

다음 통과 기준:

- generated top candidate를 실제로 듣고 piano roll로 확인한다.
- 그 후보가 one-note/two-note/chord-block/long-sustain/repeated-template 실패가 없어야 한다.

### Phase 3.7. Longer Phrase Review

목표:

- 2-bar 후보가 "만들다 만 단어"처럼 들리는 문제를 직접 검증한다.
- 같은 coverage+chord-aware 제약을 유지하되, review 후보를 `4` bar로 늘린다.
- 단순 note count 증가가 아니라 phrase로 들을 수 있는 길이와 coverage를 확보한다.

현재 결과:

- completed: `4` bar generation probe를 실행했다.
- completed: sample마다 `32` complete note groups를 생성했다.
- completed: `3/3` samples가 grammar/basic/strict gate를 통과했다.
- completed: generation probe report를 review export 입력으로 직접 사용할 수 있게 했다.
- current risk: repeated pitch ratio가 높기 때문에 motif로 들리는지 기계적 재사용으로 들리는지 확인해야 한다.

다음 통과 기준:

- exported 4-bar candidates를 piano roll과 귀로 확인한다.
- 단편이 아니라 최소한 call/continuation/landing 느낌의 phrase sketch인지 본다.
- 여전히 짧거나 기계적이면 broad generic training 전 phrase/motif-level constraint를 먼저 설계한다.

### Phase 3.8. Phrase Contour Diagnostics

목표:

- repeated pitch ratio 하나만 보고 샘플을 오판하지 않는다.
- adjacent same-note collapse, long same-pitch run, low direction change, low interval variety를 분리해서 본다.
- 후보를 자동 탈락시키기보다 review export에서 risk flag로 표시한다.

현재 결과:

- completed: generated sample report에 `phrase_contour`를 추가했다.
- completed: review export에 `risk_flags`를 추가했다.
- latest result: repeated pitch ratio는 높지만 adjacent repeated pitch ratio는 `0.000`이다.
- latest result: direction change ratio는 약 `0.689`이고 longest same pitch run은 `1`이다.

해석:

- 현재 후보는 한 음을 길게 반복하는 collapse가 아니다.
- 제한된 chord-tone pitch set을 많이 재사용하는 상태다.
- 다음 manual review는 이 pitch reuse가 motif로 들리는지, constrained cycling으로 들리는지 판단해야 한다.

### Phase 3.9. Root Bias and Tension Diagnostics

목표:

- "근음을 계속 친다"는 청취 피드백을 수치화한다.
- root tone, non-root chord tone, tension, non-chord tone 비율을 분리해서 본다.
- root collapse인지, chord-tone-only 안전함인지 판단한다.

현재 결과:

- completed: generated sample report에 `pitch_roles`를 추가했다.
- completed: review export에 `root`, `tension` columns를 추가했다.
- latest result: average root tone ratio는 약 `0.271`이다.
- latest result: top candidate root tone ratio는 약 `0.219`이다.
- latest result: tension ratio는 `0.000`이다.
- Issue #55 result: `tones_tensions`는 root tone ratio를 약 `0.271`에서 `0.135`로 낮췄고, tension ratio를 `0.000`에서 `0.313`으로 올렸다.
- Issue #55 result: 양쪽 모두 strict valid `3/3`이지만, `tones_tensions` 후보는 repeated/dominant pitch risk가 여전히 높다.
- Issue #57 result: 8-bar `approach_tensions`는 strict valid `3/3`, root ratio `0.000`, approach resolution ratio `1.000`을 만들었다.
- Issue #57 review premise: 이전보다 나아졌지만 아직 jazz solo가 아니라 다이아토닉 코드톤/근음 기반 초급 melodic exercise처럼 들린다.
- Issue #59 result: `swing_motif_approach`는 strict valid `3/3`을 유지하면서 syncopated onset ratio를 `0.500`에서 `0.750`으로 올렸다.
- Issue #59 result: unique bar-position pattern ratio는 `0.125`에서 `0.500`으로 올랐고, most-common duration ratio는 `0.552`에서 `0.380`으로 낮아졌다.
- Issue #61 result: real jazz phrase windows `57`개 기준 syncopation mean은 `0.736`, unique bar-position pattern mean은 `0.996`, duration diversity mean은 `0.379`, IOI diversity mean은 `0.341`이다.
- Issue #61 result: `swing_motif_approach`는 syncopation은 reference에 가까우나 bar-position variation, duration diversity, IOI diversity가 아직 크게 부족하다.
- Issue #63 result: real Stage B windows에서 `803`개 strictly-increasing solo-line motif를 추출했고, rhythm templates `520`, contour templates `328`, full templates `526`개를 만들었다.
- Issue #63 result: top rhythm support는 `0.009`, top contour support는 `0.012`, top full motif support는 `0.002`라서 one best motif 복붙이 아니라 distribution sampling이 필요하다.
- Issue #65 result: data-derived motif baseline도 strict `3/3`을 통과했다.
- Issue #65 result: hand-written swing 대비 bar-position variation은 `+0.500`, duration diversity는 `+0.016`, IOI diversity는 `+0.016` 개선됐지만 syncopation은 `-0.125` 낮아졌다.
- Issue #67 result: `data_motif`와 `hand_written_swing` 후보를 mode/sample/rank가 드러나는 named MIDI review package로 export했다.
- Issue #69 result: chord/bass guide가 들어간 context MIDI와 straight-grid timing reference를 추가했다.
- Issue #71 result: `straight_guide_tones` 후보를 추가해 swing timing 문제와 chromatic/scale pitch 문제를 분리했다.
- Issue #71 result: `straight_guide_tones`는 note count `64`, unique pitch count `26-29`, chord-tone ratio `0.656`, tension ratio `0.172`, root-tone ratio `0.000`이지만 straight reference용 dead-air gate 때문에 strict `0/3`이다.
- Issue #73 result: `data_motif_guide_tones` 후보를 추가해 data-derived rhythm/duration template과 guide-tone/cadence pitch grammar를 결합했다.
- Issue #73 result: `data_motif_guide_tones`는 strict `3/3`, note count `63`, unique pitch count `23-24`, chord-tone ratio `0.797`, tension ratio `0.062`, root-tone ratio `0.000`, unique bar-position pattern ratio `1.000`이다.
- Issue #75 result: reference pitch-role landing 통계를 시도했지만 known chord note ratio가 `0.000`이라 pitch-role reference는 아직 사용할 수 없다.
- Issue #75 result: 현재 비교 가능한 것은 rhythm reference뿐이며, pitch vocabulary 조정 전에 chord annotation coverage audit이 필요하다.
- Issue #77 result: role metadata `2812`개, raw sidecar `0`개, text event를 검사한 MIDI file `120`개를 scan했지만 chord progression hit는 `0`이다.
- Issue #77 result: 현재 local dataset에는 바로 쓸 수 있는 chord progression annotation이 없으므로 reference pitch-role comparison은 아직 불가능하다.
- Issue #79 result: `inline_notes` tiny fixture `2` samples, `32` notes로 chord-labeled eval contract를 검증했다.
- Issue #79 result: fixture chord-tone ratio는 `0.844`, tension ratio는 `0.156`, outside ratio는 `0.000`이다.
- Issue #81 result: generated candidate bridge fixture `1` sample, `16` notes로 report-to-evaluator 연결을 검증했다.
- Issue #81 result: fixture chord-tone ratio는 `1.000`, tension ratio는 `0.000`, outside ratio는 `0.000`이다.
- Issue #83 result: data-guide hybrid generated chord eval은 `6` candidates, `192` notes를 평가했다.
- Issue #83 result: aggregate chord-tone ratio는 `0.656`, tension ratio는 `0.120`, outside ratio는 `0.000`이다.
- Issue #83 result: `data_motif` chord-tone ratio는 `0.500`, `data_motif_guide_tones` chord-tone ratio는 `0.812`이다.
- Issue #85 result: combined review markdown is written to `outputs/stage_b_generated_chord_eval/harness_stage_b_review_markdown_chord_eval/review_candidates_with_chord_eval.md`.
- Issue #87 result: listening review notes template contains `6` pending candidates and validates phrase quality, timing, chord fit, issue flags, and decision enums.
- Issue #89 result: listening review aggregate reports `6` pending candidates, `0` reviewed candidates, and only recommends `collect_listening_reviews`.
- Issue #91 result: full review manifest notes contain `15` pending candidates with `review_midi_path`, `context_midi_path`, mode, rank, sample, and rhythm/timing metrics.
- Issue #93 result: objective MIDI review flags `chromatic_walk=7`, `duration_pattern_collapse=9`, `overlap_polyphonic=9`, and `too_stepwise_or_scalar=4`.
- Issue #95 result: objective review priority reports `15` candidates, `6` warning/reviewable candidates, and `9` problem candidates before subjective listening.
- Issue #97 result: overlap-free review export reports `15` reviewable candidates, `5` clean candidates, `10` warning candidates, and `overlap_polyphonic=0`.
- Issue #99 result: duration variation review reports `15` reviewable candidates, `8` clean candidates, `7` warning candidates, `duration_pattern_collapse=0`, and `overlap_polyphonic=0`.
- Issue #101 result: phrase/cadence review reports `12` reviewable candidates, `11` clean candidates, `1` warning candidate, `chromatic_walk=1`, and `too_stepwise_or_scalar=0`.
- Issue #103 result: phrase naturalness review reclassifies the same `12` candidates as warnings because `unresolved_large_leaps=12`.
- Issue #105 result: phrase recovery review reports `phrase_cadence` candidates as `3` warnings and `phrase_recovery` candidates as `3` clean candidates.
- Issue #107 result: data motif phrase recovery review reports `data_motif_guide_tones` as `3` warnings and `data_motif_phrase_recovery` as `3` clean candidates.
- Issue #109 result: objective clean review package keeps only the `3` `data_motif_phrase_recovery` candidates and writes `outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md`.
- Issue #111 result: clean context diagnostics reports `3` candidates, diagnostic flags `{}`, bar coverage `8/8`, off-grid ratio `0.000`, max duration `1.000` beat, and decision hint `listen_with_context`.
- Issue #113 result: clean listening review notes template covers the `3` objective-clean context candidates and validates review enums/summary output.
- 2026-05-24 MIDI-note proxy review result: `needs_followup=2`, `reject=1`, `keep=0`; the strongest candidate is still `timing=stiff`, `jazz_vocabulary=thin`.
- Issue #115 result: `data_motif_contour_landing_repair` is strict `3/3`, final landing resolved `3/3`, max interval `7`, abrupt resets `0`, and objective MIDI flag counts `{}`.
- Issue #115 comparison: `data_motif_phrase_recovery` is still strict `3/3`, but final landing resolved is `1/3` and max interval is `13`.
- Issue #116 contour repair MIDI-note proxy review result: `reviewed=6`, `needs_followup=5`, `reject=1`, `keep=0`.
- Issue #116 aggregate result: `phrase=1`, `fragment=4`, `exercise=1`, `too_stiff=6`, `fits=4`, `unclear=2`.
- Issue #118 result: `data_motif_rhythm_phrase_variation` is strict `3/3`, final landing resolved `3/3`, max interval `6`, objective flags `{}`, and pitch range floor `>=51`.
- Issue #118 rhythm result: syncopation `0.694`, duration diversity `0.097`, IOI diversity `0.115`, compared with contour repair `0.625`, `0.062`, `0.079`.
- Issue #120 MIDI-note proxy review result: `reviewed=6`, `needs_followup=4`, `reject=2`, `keep=0`.
- Issue #120 result: variation rank 1 is the representative follow-up candidate, but variation rank 2 and rank 3 are exact duplicates of rank 1 and should not be treated as independent listening evidence.
- Issue #122 result: review export now reports `unique_note_sequence_count=6` and `duplicate_note_sequence_count=0`.
- Issue #122 result: the repaired variation candidates remain strict `3/3`, final landing `3/3`, max interval `6`, and objective flags `{}`.
- Issue #124 MIDI-note proxy review result: `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`.
- Issue #124 aggregate result: `improve_phrase_vocabulary=14`, `fix_timing_grid=12`, `increase_motif_variation=6`.
- Issue #126 timing-grid repair result: variation `avg_most_common_ioi_ratio=0.412`, `max_interval=4`, `duplicate_note_sequence_count=0`, objective flags `{}`.
- Issue #126 tradeoff: variation `avg_ioi_diversity_ratio=0.070`, `avg_unique_bar_position_pattern_ratio=0.583`, and `avg_duration_diversity_ratio=0.084`.
- Issue #128 MIDI-note proxy review result: `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`, and `too_stiff=6`.
- Issue #128 aggregate result: `improve_phrase_vocabulary=14`, `fix_timing_grid=12`, `increase_motif_variation=6`.
- Issue #128 confirms timing repair should be kept as a guardrail but not treated as a musical solution; the next generation bottleneck is phrase-vocabulary diversity.
- Issue #130 phrase-vocabulary repair result: variation `avg_unique_bar_position_pattern_ratio=0.958`, `avg_ioi_diversity_ratio=0.091`, `avg_most_common_ioi_ratio=0.385`, `avg_most_common_duration_ratio=0.384`, and objective flags `{}`.
- Issue #130 variation review MIDI has duplicate note sequence count `0` and before/after max simultaneous notes `1/1`.
- Issue #132 MIDI-note proxy review result: `reviewed=6`, `needs_followup=6`, `reject=0`, `keep=0`, timing `acceptable=2`, `too_stiff=4`.
- Issue #132 aggregate result: `improve_phrase_vocabulary=11`, `fix_timing_grid=8`, `increase_motif_variation=5`, `increase_tension_approach_vocabulary=2`.
- Issue #134 phrase-shape/tension repair result: variation `avg_tension_ratio=0.437`, objective tension ratio `0.500-0.540`, duplicate note sequences `0`, before/after max simultaneous notes `1/1`, and objective flags `{}`.
- Issue #136 MIDI-note proxy review result: `reviewed=6`, `keep=1`, `needs_followup=5`, `reject=0`, timing `acceptable=2`, `too_stiff=4`, chord fit `fits=6`.
- Issue #136 aggregate result: `improve_phrase_vocabulary=10`, `fix_timing_grid=8`, `increase_motif_variation=5`, `increase_tension_approach_vocabulary=0`.

해석:

- 현재 후보는 root-only collapse가 아니다.
- 오히려 `chord_pitch_mode=tones` 때문에 tension이 전혀 없는 안전한 chord-tone-only line이다.
- `tones_tensions`는 no-tension 문제를 줄였지만, 더 좋은 solo phrase라고 바로 판단할 단계는 아니다.
- `approach_tensions`는 pitch-level resolution을 만들지만, 이 또한 jazz vocabulary 자체는 아니다.
- `swing_motif_approach`는 기계적인 grid 반복을 줄였지만, 이 또한 jazz vocabulary 자체는 아니다.
- real phrase reference stats와 motif extraction 기준으로 보면 다음은 hand-written rhythm rule 확장이 아니라 data-derived motif/cadence control 쪽이 맞다.
- 다만 pitch-role 쪽은 real reference chord label이 아직 없으므로, 다음 개선은 실제 청취 결과를 notes에 채운 뒤 issue distribution으로 후속 generation rule을 분기하는 순서가 맞다.
- clean package의 context MIDI review boundary는 proxy review까지 진행됐다.
- proxy review는 실제 오디오 청취가 아니므로 최종 subjective quality proof가 아니다.
- Issue #115는 contour continuity와 final landing objective target을 개선했다.
- contour repair MIDI-note proxy review 결과, 다음 병목은 landing이 아니라 rhythm stiffness, repeated duration/rest template, thin phrase vocabulary다.
- Issue #118은 그 병목 중 rhythm objective metrics와 register floor를 개선했다.
- Issue #120 proxy review 결과, 다음 병목은 exact duplicate rank candidates를 없애는 sample diversity repair였다.
- Issue #122는 duplicate 문제를 고쳤다.
- Issue #124는 sample-diverse 후보도 여전히 `too_stiff=6`임을 확인했으므로, 다음 병목은 timing-grid repetition repair다.
- Issue #126은 timing-grid repetition을 줄였지만 diversity tradeoff가 있으므로, 다음은 repaired candidates proxy review다.
- Issue #128은 repaired candidates도 no-keep임을 확인했으므로, 다음은 objective-clean/duplicate-free 조건을 유지한 phrase-vocabulary diversity repair다.
- Issue #130은 objective surface를 개선했으므로, 다음은 repaired candidates가 proxy review에서 `too_stiff`를 줄였는지 확인하는 review issue다.
- Issue #132는 phrase-vocabulary repair가 timing proxy를 개선했지만 no-keep이므로, 다음은 phrase shape와 tension/approach vocabulary repair다.
- Issue #134는 phrase-shape/tension objective surface를 개선했으므로, 다음은 repaired candidates가 proxy review에서 `too_safe`와 phrase sketch/exercise 문제를 줄였는지 확인하는 review issue다.
- Issue #136은 첫 proxy keep 후보를 확인했으므로, 다음은 해당 후보를 focused context review package로 격리하는 issue다.

### Phase 3.10. Swing/Motif Phrase Grammar

목표:

- pitch-only approach/tension constraint의 한계를 확인한다.
- 같은 checkpoint에서 baseline approach grammar와 swing/motif rhythm grammar를 비교한다.
- rhythm profile을 candidate ranking과 review export에 넣는다.

현재 결과:

- completed: `jazz_rhythm_position_tokens()`와 `jazz_rhythm_duration_tokens()`를 추가했다.
- completed: `approach_baseline`과 `swing_motif_approach`를 같은 checkpoint에서 비교했다.
- latest result: 두 grammar 모두 strict valid `3/3`이다.
- latest result: syncopated onset ratio는 `0.500`에서 `0.750`으로 좋아졌다.
- latest result: unique bar-position pattern ratio는 `0.125`에서 `0.500`으로 좋아졌다.
- latest result: direct MIDI inspection에서 baseline의 반복 IOI/template 문제가 확인됐다.

해석:

- 현재 후보는 one-note/two-note/chord-block failure가 아니다.
- rhythmic template 반복은 줄었다.
- 하지만 아직 실제 jazz solo vocabulary라고 볼 근거는 부족하다.

다음 통과 기준:

- generated rhythm profile을 real jazz MIDI window 통계와 비교한다.
- pitch motif cell, cadence/landing, phrase memory 중 하나를 다음 issue로 분리한다.
- rule이 아니라 data-derived constraint로 넘어갈지 판단한다.

### Phase 3.11. Real Phrase Reference Statistics

목표:

- generated MIDI가 "이전보다 나음"인지 "실제 jazz phrase 통계에 가까움"인지 분리한다.
- real Stage B phrase windows에서 rhythm/contour reference metrics를 만든다.
- generated candidate report와 comparable metric key를 맞춘다.

현재 결과:

- completed: `scripts/run_stage_b_reference_stats.py`를 추가했다.
- completed: `4`개 MIDI 파일에서 `57`개 8-bar real phrase windows를 분석했다.
- latest result: real syncopated onset ratio mean은 `0.736`이다.
- latest result: real unique bar-position pattern ratio mean은 `0.996`이다.
- latest result: real duration diversity ratio mean은 `0.379`이다.
- latest result: real IOI diversity ratio mean은 `0.341`이다.
- latest result: Issue #59 `swing_motif_approach`는 syncopation은 reference와 거의 맞지만 bar-position/duration/IOI diversity는 부족하다.

해석:

- Issue #59는 baseline보다 나아졌지만 아직 real jazz window 통계에는 미달한다.
- 특히 every-bar pattern variation이 부족하다.
- 다음 단계는 hand-written swing pattern을 더 추가하기보다 dataset에서 phrase motif templates를 추출하는 것이다.

다음 통과 기준:

- real window에서 rhythm/motif templates를 추출한다.
- generated candidate가 reference p25-p75 범위 안에 들어오는 metric을 늘린다.
- phrase ending/cadence도 reference 기반으로 비교한다.

### Phase 3.12. Data-Derived Motif Template Extraction

목표:

- hand-written swing/motif rule을 더 늘리지 않는다.
- real Stage B phrase windows에서 rhythm, contour, full motif templates를 추출한다.
- chord-block 또는 same-onset voicing이 solo-line motif catalog를 오염시키지 않도록 기본 필터를 둔다.
- 다음 generation probe가 data-derived rhythm/contour distribution을 사용할 수 있게 만든다.

현재 결과:

- completed: `scripts/run_stage_b_motif_template_extraction.py`를 추가했다.
- completed: same-onset/non-increasing onset motif를 기본적으로 제외한다.
- completed: `4`개 MIDI 파일에서 만든 Stage B 8-bar windows 기준 `803`개 motif를 추출했다.
- latest result: source records `56`, rhythm templates `520`, contour templates `328`, full templates `526`.
- latest result: top full motif support가 `0.002`라서 full motif를 그대로 복사하는 방식은 맞지 않다.

해석:

- 실제 jazz phrase material은 매우 분산되어 있다.
- 다음 단계는 top motif 하나를 쓰는 것이 아니라 rhythm template과 contour template을 분리해서 sampling하는 것이다.
- 이 단계는 생성 품질을 바로 올리는 작업이 아니라, beginner-like hand-written line에서 data-derived phrase material로 넘어가는 준비다.

다음 통과 기준:

- data-derived motif catalog를 constrained generation의 position/duration/contour 후보로 연결한다.
- generated candidate의 duration diversity와 IOI diversity가 Issue #59보다 좋아지는지 본다.
- reference p25-p75 범위에 가까워지는 metric을 늘린다.
- piano roll에서 chord-tone 나열이 아니라 phrase contour로 들리는지 review export로 확인한다.

### Phase 3.13. Data-Derived Motif Baseline Generation

목표:

- Issue #63 motif catalog를 실제 8-bar generation baseline에 연결한다.
- hand-written `swing_motif_approach`와 data-derived motif baseline을 같은 조건에서 비교한다.
- 생성이 strict gate를 통과하는지, rhythm diversity가 나아지는지 본다.

현재 결과:

- completed: `scripts/run_stage_b_data_motif_generation_compare.py`를 추가했다.
- completed: extracted rhythm template을 position/duration 후보로 사용한다.
- completed: extracted contour template을 pitch interval 후보로 사용한다.
- completed: duration을 다음 onset 전까지 제한해 overlap/postprocess removal을 막는다.
- latest result: `hand_written_swing` strict `3/3`, `data_motif` strict `3/3`.
- latest result: data-derived baseline은 bar-position variation을 `0.500`에서 `1.000`으로 올렸다.
- latest result: duration repetition ratio를 `0.750`에서 `0.375`로 낮췄다.
- latest result: syncopation은 `0.750`에서 `0.625`로 낮아졌다.

해석:

- data-derived motif baseline은 hand-written baseline보다 pattern variation 면에서 낫다.
- 하지만 syncopation 하락과 낮은 diversity 상승폭 때문에 바로 더 좋은 jazz solo라고 말할 수 없다.
- 다음은 MIDI review export와 listening/piano-roll 비교가 필요하다.

다음 통과 기준:

- data_motif 후보를 hand_written_swing 후보와 파일명으로 구분해 review export한다.
- 실제 piano roll에서 phrase contour가 초급 scale exercise보다 나은지 확인한다.
- data_motif가 review 가치가 있으면 model constrained generation 쪽에 연결하고, 아니면 contour/cadence extraction을 더 강화한다.

### Phase 3.14. Data Motif Review Export

목표:

- `data_motif`와 `hand_written_swing` 후보를 mode/sample/rank 기준으로 구분한다.
- piano-roll/listening review가 가능하도록 named MIDI package를 만든다.
- review markdown에 핵심 metric을 같이 남긴다.

현재 결과:

- completed: `review_manifest.json`과 `review_candidates.md`를 생성한다.
- completed: `named_midi/` 아래에 mode가 드러나는 MIDI 파일명을 만든다.
- latest result: review candidates `6`.
- latest result: `data_motif` strict `3/3`, `hand_written_swing` strict `3/3`.

해석:

- 이제 숫자 비교가 아니라 실제 청취 리뷰가 가능하다.
- syncopation 하락이 체감상 나쁜지, duration repetition 감소가 실제로 더 나은지 확인해야 한다.

다음 통과 기준:

- named MIDI 후보를 직접 듣고 piano roll로 확인한다.
- data_motif가 더 자연스러우면 motif sampling을 model constrained generation에 연결한다.
- 둘 다 초급 scale exercise면 cadence/phrase-ending extraction을 먼저 강화한다.

### Phase 3.15. Chord Context and Straight-Grid Review

목표:

- solo-only MIDI 리뷰의 한계를 줄인다.
- chord/bass guide가 포함된 context MIDI를 생성한다.
- swing/motif timing이 문제인지 확인하기 위해 straight-grid reference를 같이 export한다.

현재 결과:

- completed: `chord_guide.mid`를 생성한다.
- completed: candidate별 `*_with_context.mid`를 생성한다.
- completed: `straight_grid` baseline mode를 추가했다.
- latest result: `data_motif` strict `3/3`, `hand_written_swing` strict `3/3`.
- latest result: `straight_grid`는 timing reference로 export한다.

해석:

- 이제 chord progression 위에서 line이 in인지 out인지 들을 수 있다.
- swing/motif가 musical swing이 아니라 timing drift처럼 들리는지 비교할 수 있다.
- straight_grid는 더 좋은 솔로가 아니라 timing 기준점이다.

다음 통과 기준:

- context MIDI를 직접 듣고 `data_motif`, `hand_written_swing`, `straight_grid`를 비교한다.
- swing이 거슬리면 generated output은 straight quantized grid를 기본으로 둔다.
- chord context 위에서도 phrase가 초급스럽다면 cadence/phrase-ending extraction을 먼저 강화한다.

### Phase 4. Generic Jazz Base 후보 학습

목표:

- Brad-only scratch training이 아니라 generic jazz pianist prior를 만든다.

조건:

- Stage B 2-file probe가 최소한 reviewable MIDI를 만든 뒤에만 진행한다.
- dataset audit 결과를 사용해 non-Brad generic jazz split을 만든다.
- Brad subset은 adaptation/holdout으로 분리한다.

통과 기준:

- generic split에서 train/val leakage가 없다.
- broad training 결과가 Brad-only tiny probe보다 안정적이다.
- generated MIDI가 여러 sample에서 review gate를 통과한다.

현재 readiness audit:

- Issue #385 result: dataset readable `2777`, non-Brad candidate `2703`, Brad holdout `72`, duplicate exact hash groups `0`
- Stage B objective path: `outside_soloing_repair_objective_path_complete`
- phase4 prep ready: `true`
- broad training execution ready: `false`
- broad trained-model quality / Brad style adaptation claim: `false`
- 다음 작업은 broad training 실행이 아니라 Stage B generic train/val manifest contract 갱신이다.

현재 manifest contract:

- Issue #387 result: generic_jazz_train `2433`, generic_jazz_val `270`, brad_adaptation_train `47`, brad_adaptation_val `11`, brad_test_holdout `14`
- non-Brad split count: expected `2703`, actual `2703`
- Brad split count: expected `72`, actual `72`
- leakage/overlap: `0`
- manifest contract ready: `true`
- broad training execution ready: `false`
- 다음 작업은 generic split manifest를 사용한 Stage B duration-explicit window preparation smoke다.

현재 generic window smoke:

- Issue #389 result: selected train/val files `6/3`
- tokenized train/val records: `556/191`
- max token id / vocab size: `544/547`
- fits vocab: `true`
- Stage B window prepare smoke ready: `true`
- generic base training execution ready: `false`
- 다음 작업은 generic base tiny training smoke다.

현재 generic tiny training smoke:

- Issue #391 result: selected train/val records `32/8`
- token files: `40`
- max token id / vocab size: `544/547`
- training returncode: `0`
- best validation loss: `6.1427`
- tiny training smoke passed: `true`
- broad trained-model quality / Brad style adaptation claim: `false`
- 다음 작업은 tiny checkpoint generation probe다.

현재 generic tiny checkpoint generation probe:

- Issue #393 result: generation command returncode `0`
- sample count: `2`
- valid / strict / grammar gate sample count: `0/0/0`
- collapse warning sample rate: `0.5`
- avg onset / sustained coverage ratio: `0.046875/0.09375`
- diagnostic failure reasons: `note count too low: 4 < 6`, `note count too low: 3 < 6; collapse=single_pitch,single_position`
- generation/decode/report path executable: `true`
- raw generation quality ready: `false`
- broad trained-model quality / Brad style adaptation claim: `false`
- 다음 작업은 tiny checkpoint grammar repair다.

현재 generic tiny checkpoint grammar repair:

- Issue #395 result: baseline valid/strict/grammar `0/0/0`
- repair valid/strict/grammar: `2/2/2`
- grammar / valid / strict delta: `2/2/2`
- repair collapse warning sample rate: `0.0`
- repair avg postprocess removal ratio: `0.125`
- repair avg onset / sustained coverage ratio: `0.1875/0.375`
- raw generation quality / constrained generation quality claim: `false/false`
- broad trained-model quality / Brad style adaptation claim: `false`
- 다음 작업은 repair repeatability probe다.

현재 generic tiny checkpoint repair repeatability:

- Issue #397 result: sample count `6`
- valid / strict / grammar gate sample count: `5/5/6`
- valid / strict / grammar rate: `0.8333333333333334/0.8333333333333334/1.0`
- collapse warning sample rate: `0.0`
- avg postprocess removal ratio: `0.08333333333333333`
- failure reason: `dead-air ratio too high: 1.000 >= 0.800` `1`
- raw generation quality / constrained generation quality claim: `false/false`
- broad trained-model quality / Brad style adaptation claim: `false`
- 다음 작업은 repair review package다.

현재 generic tiny checkpoint repair review package:

- Issue #399 result: source sample count `6`
- strict-valid review candidates: `5`
- failed candidate count: `1`
- rank 1: seed `47`, sample `6`, dead-air `0.5`, coverage `0.6562486875`
- rank 2: seed `45`, sample `4`, dead-air `0.5714285714285714`, coverage `0.8437483124999999`
- rank 3: seed `42`, sample `1`, dead-air `0.6666666666666666`, coverage `0.9062481875`
- failed row: seed `44`, sample `3`, reason `dead-air ratio too high: 1.000 >= 0.800`
- musical quality / broad trained-model quality / Brad style adaptation claim: `false/false/false`
- 다음 작업은 repair listening notes다.

현재 generic tiny checkpoint repair listening notes:

- Issue #401 result: source candidate count `5`
- notes candidate count: `5`
- notes status: `pending_human_review`
- human review filled: `false`
- musical quality / broad trained-model quality / Brad style adaptation claim: `false/false/false`
- 다음 작업은 repair listening fill이다.

현재 generic tiny checkpoint repair listening fill:

- Issue #403 result: review input present `false`
- fill status: `pending_review_input`
- listening fill status: `pending_review_input`
- candidate count / keep count: `5/0`
- human review filled: `false`
- musical quality / broad trained-model quality / Brad style adaptation claim: `false/false/false`
- objective-only auto progress allowed: `true`
- 다음 작업은 repair audio render package다.

현재 generic tiny checkpoint repair audio render package:

- Issue #405 result: planned audio outputs `5`
- render status: `ready_for_local_render`
- selected renderer: `fluidsynth`
- soundfont exists: `true`
- render attempted: `false`
- audio rendered quality / human audio preference / musical quality claim: `false/false/false`
- 다음 작업은 repair local audio render attempt다.

현재 generic tiny checkpoint repair local audio render attempt:

- Issue #407 result: rendered audio files `5`
- technical WAV validation: `true`
- sample rate: `44100`
- duration seconds range: `7.766-10.657`
- audio rendered quality / human audio preference / musical quality claim: `false/false/false`
- 다음 작업은 repair user listening review input이다.

현재 generic tiny checkpoint repair user listening review:

- Issue #409 result: reviewed audio files `5`
- overall decision: `reject_all`
- candidate decision: `reject`
- primary failure: `plunk_and_stop`
- timing / phrase / vocabulary: `too_short_or_stiff` / `fragmented` / `not_musical`
- human/audio keep claim: `false`
- 다음 작업은 repair phrase continuation decision이다.

현재 generic tiny checkpoint repair phrase continuation decision:

- Issue #411 result: input boundary `generic_tiny_checkpoint_repair_audio_review_reject_all`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep`
- repair target count: `6`
- auto progress allowed: `true`
- human/audio keep / musical quality / broad model quality claim: `false/false/false`
- 다음 작업은 repair phrase continuation sweep이다.

현재 generic tiny checkpoint repair phrase continuation sweep:

- Issue #413 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package`
- sample count: `6`
- valid / strict / grammar: `3/1/6`
- target qualified count: `1`
- selected objective candidate: sample `1`, seed `62`
- selected note count / coverage / tail empty: `9` / `0.9062481875` / `2`
- selected chord-role ratio / postprocess removal: `0.5625` / `0.4375`
- musical quality / broad model quality claim: `false/false`
- 다음 작업은 repair phrase continuation audio render package다.

현재 generic tiny checkpoint repair phrase continuation audio render package:

- Issue #415 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt`
- render status: `ready_for_local_render`
- selected renderer: `fluidsynth`
- soundfont exists: `true`
- planned audio outputs: `1`
- selected objective candidate: sample `1`, seed `62`
- render attempted: `false`
- audio rendered quality / human audio preference claim: `false/false`
- 다음 작업은 repair phrase continuation local audio render attempt다.

현재 generic tiny checkpoint repair phrase continuation local audio render attempt:

- Issue #417 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_user_listening_review_input`
- rendered audio file count: `1`
- technical WAV validation: `true`
- sample rate: `44100`
- duration seconds: `9.326`
- size bytes: `1645100`
- audio rendered quality / human audio preference claim: `false/false`
- 다음 작업은 repair phrase continuation user listening review input이다.

현재 generic tiny checkpoint repair phrase continuation MIDI note failure review:

- Issue #419 result: boundary `generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision`
- overall decision: `reject_all`
- primary failure: `midi_note_random_large_leaps`
- note count: `9`
- pitch range: `29-89`
- pitch span: `60`
- max abs interval: `60`
- interval sequence: `[15, -24, 60, -60, 34, -3, 27, -34]`
- large interval ratio: `0.875`
- severe interval count: `6`
- musical quality / human audio keep claim: `false/false`
- 다음 작업은 repair phrase continuation range interval guard decision이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard decision:

- Issue #421 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep`
- observed pitch span / target: `60` / `24`
- observed max abs interval / target: `60` / `12`
- observed large interval ratio / target: `0.875` / `0.35`
- observed severe interval count / target: `6` / `0`
- preferred pitch range: `48-84`
- repair target count: `5`
- musical quality claim: `false`
- 다음 작업은 repair phrase continuation range interval guard sweep이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sweep:

- Issue #423 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package`
- interval cap sweep: `12/9/7/5`
- target qualified: `3/48`
- top candidate: interval cap `9`, sample seed `70`, sample `9`
- top note count / phrase coverage / tail empty: `11` / `1.0` / `0`
- top pitch span / max abs interval / large interval ratio: `21` / `9` / `0.0`
- quality claim: `false`
- 다음 작업은 repair phrase continuation range interval guard audio render package다.

현재 generic tiny checkpoint repair phrase continuation range interval guard audio render package:

- Issue #425 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt`
- render status: `ready_for_local_render`
- selected renderer / soundfont exists: `fluidsynth` / `true`
- planned audio outputs: `3`
- target-qualified ranks: `1-3`
- render attempted: `false`
- audio quality / human preference / musical quality claim: `false`
- 다음 작업은 repair phrase continuation range interval guard local audio render attempt다.

현재 generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt:

- Issue #427 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review_input`
- rendered audio files: `3`
- technical WAV validation: `true`
- sample rate: `44100`
- duration range: `6.818s-7.194s`
- audio quality / human preference / musical quality claim: `false`
- critical user input required: `true`
- 다음 작업은 user listening review input이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard user listening review:

- Issue #429 result: boundary `generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_review_reject_all`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis`
- reviewed audio files: `3`
- overall decision / candidate decision: `reject_all` / `reject`
- primary failure: `subjective_not_musical`
- timing / phrase / vocabulary: `outside_or_unclear` / `not_musical` / `not_musical`
- human audio keep / musical quality claim: `false` / `false`
- 다음 작업은 range interval guard rejection analysis다.

현재 generic tiny checkpoint repair phrase continuation range interval guard rejection analysis:

- Issue #431 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis`
- source boundary: `generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_review_reject_all`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision`
- analyzed candidates: `3`
- common evidence flag: `high_dead_air_or_sparse_phrase`
- evidence flag counts: `high_dead_air_or_sparse_phrase=3`, `long_internal_gap_present=2`, `octave_or_larger_interval_present=2`, `adjacent_pitch_repeat_present=2`
- primary next repair target: `sparse_phrase_continuity_after_range_interval_guard`
- quality root cause / musical quality claim: `false` / `false`
- 다음 작업은 sparse phrase repair decision이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision:

- Issue #433 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision`
- source boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep`
- observed gap ratio max: `0.5312`
- observed max internal gap max: `1.5`
- target max gap ratio / max internal gap: `0.4` / `0.75`
- target min note count / min phrase coverage: `10` / `0.9`
- primary repair target: `sparse_phrase_continuity_after_range_interval_guard`
- quality root cause / musical quality claim: `false` / `false`
- 다음 작업은 sparse phrase repair sweep이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep:

- Issue #435 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package`
- target passed: `true`
- target qualified count / candidate count: `18` / `24`
- top candidate: interval cap `5`, sample seed `86`, sample `7`
- top note count / max abs interval: `12` / `8`
- top gap ratio / source max: `0.2188` / `0.5312`
- top max internal gap / source max: `0.5` / `1.5`
- tail_empty decision target `0`은 top 후보에서 `1`로 남아 soft failure 기록
- human/audio preference 및 musical quality claim: `false`
- 다음 작업은 sparse phrase audio render package다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package:

- Issue #437 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package`
- status: `ready_for_local_render`
- next boundary: `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt`
- planned audio outputs: `3`
- selected renderer: `/opt/homebrew/bin/fluidsynth`
- soundfont exists: `true`
- review ranks: cap/seed/sample `5/86/7`, `5/80/1`, `9/86/7`
- audio output / audio quality / human preference claim: `false` / `false` / `false`
- 다음 작업은 sparse phrase local audio render attempt다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt:

- Issue #439 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt`
- rendered audio files: `3`
- technical WAV validation: `true`
- sample rate: `44100`
- duration range: `6.792s-7.094s`
- rank 1: `rank_01_cap_5_seed_86_sample_7.wav`
- rank 2: `rank_02_cap_5_seed_80_sample_1.wav`
- rank 3: `rank_03_cap_9_seed_86_sample_7.wav`
- audio rendered quality / human preference / musical quality claim: `false` / `false` / `false`
- 다음 작업은 sparse phrase user listening review input이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review:

- Issue #441 result: boundary `generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_review_reject_all`
- reviewed audio files: `3`
- overall decision: `reject_all`
- candidate decision: `reject`
- primary failure: `subjective_not_musical`
- timing / phrase / vocabulary: `outside_or_unclear` / `not_musical` / `not_musical`
- human/audio keep claimed: `false`
- musical quality claim: `false`
- broad trained model quality claim: `false`
- 다음 작업은 sparse phrase rejection analysis다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis:

- Issue #443 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis`
- analyzed candidates: `3`
- candidates without objective evidence flags: `1`
- objective proxy gap recorded: `true`
- common evidence flags: 없음
- primary next review target: `model_core_review_after_objective_proxy_gap`
- musical quality / quality cause claim: `false` / `false`
- 판단: 추가 후처리 규칙 반복보다 model core, dataset, training boundary 검토 필요
- 다음 작업은 sparse phrase model core review decision이다.

현재 generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review decision:

- Issue #445 result: boundary `stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision`
- decision: `stop_constraint_postprocess_repair_loop`
- continue constraint/postprocess repair loop: `false`
- tiny checkpoint role: `diagnostic_only`
- model core transition required: `true`
- objective proxy gap recorded: `true`
- candidate without objective flags: `1`
- musical quality / broad trained model quality claim: `false` / `false`
- 다음 작업은 generic model-core training data plan이다.

현재 generic model-core training data plan:

- Issue #447 result: boundary `stage_b_generic_model_core_training_data_plan`
- repair loop status: `stopped`
- tiny checkpoint role: `diagnostic_only`
- generic train / val files: `2433` / `270`
- Brad split: `47` / `11` / `14`
- window smoke token max / vocab: `544` / `547`
- tiny training selected records: `32` / `8`
- full window preparation / full training executed: `false` / `false`
- broad trained model quality claim: `false`
- 다음 작업은 generic full manifest window preparation이다.

현재 generic full manifest window preparation:

- Issue #449 result: boundary `stage_b_generic_full_manifest_window_preparation`
- train / val manifest files: `2433` / `270`
- generated samples: `175981`
- tokenized train / val files: `154136` / `21845`
- max token id / vocab size: `544` / `547`
- fits vocab: `true`
- full training executed: `false`
- broad trained model quality claim: `false`
- output size: 약 `2.7GB`
- 다음 작업은 generic base training scale smoke다.

현재 generic base training scale smoke:

- Issue #451 result: boundary `stage_b_generic_base_training_scale_smoke`
- source tokenized train / val files: `154136` / `21845`
- selected train / val records: `128` / `32`
- token files: `160`
- max token id / vocab size: `544` / `547`
- fits vocab: `true`
- training returncode: `0`
- best validation loss: `5.9031`
- checkpoint count: `1`
- full generic training executed: `false`
- broad trained model quality / Brad style adaptation claim: `false` / `false`
- 다음 작업은 generic base scale checkpoint generation probe다.

현재 generic base scale checkpoint generation probe:

- Issue #453 result: boundary `stage_b_generic_base_scale_checkpoint_generation_probe`
- generation command returncode: `0`
- sample count: `3`
- valid / strict / grammar gate sample count: `0` / `0` / `0`
- collapse warning sample rate: `0.0`
- avg onset / sustained coverage ratio: `0.0625` / `0.09375`
- max longest sustained empty run steps: `25`
- failure reasons: `note count too low: 4 < 6`, `3 < 6`, `2 < 6`
- raw generation quality ready: `false`
- broad trained model quality / Brad style adaptation claim: `false` / `false`
- 다음 작업은 generic base scale checkpoint grammar representation decision이다.

현재 generic base scale checkpoint grammar representation decision:

- Issue #455 result: boundary `stage_b_generic_base_scale_checkpoint_grammar_representation_decision`
- decision: `select_density_coverage_repair_probe`
- selected target: `target_density_coverage_repair`
- sample count: `3`
- valid / strict / grammar gate sample count: `0` / `0` / `0`
- note count failure count: `3`
- all samples note-count failed: `true`
- avg onset / sustained coverage ratio: `0.0625` / `0.09375`
- collapse warning not primary: `true`
- postprocess-only repair / audio review selected: `false` / `false`
- quality root cause / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint density coverage repair probe다.

현재 generic base scale checkpoint density coverage repair probe:

- Issue #457 result: boundary `stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe`
- baseline valid / strict / grammar gate: `0` / `0` / `0`
- repair valid / strict / grammar gate: `1` / `1` / `3`
- baseline / repair note count failure count: `3` / `0`
- note count failure delta: `3`
- baseline avg onset / sustained coverage: `0.0625` / `0.09375`
- repair avg onset / sustained coverage: `0.16666666666666666` / `0.6354166666666666`
- onset / sustained coverage delta: `0.10416666666666666` / `0.5416666666666666`
- density/coverage target qualified: `true`
- remaining failure reason: `too many long notes: 0.333 > 0.250` `2`
- raw generation quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint density coverage remaining blocker decision이다.

현재 generic base scale checkpoint density coverage remaining blocker decision:

- Issue #459 result: boundary `stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision`
- decision: `select_duration_long_note_repair_probe`
- selected target: `duration_long_note_ratio_repair`
- remaining blocker: `duration_long_note_ratio`
- valid / strict / grammar gate sample count: `1` / `1` / `3`
- long-note failure count: `2`
- audio review selected: `false`
- musical quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint duration long-note repair probe다.

현재 generic base scale checkpoint duration long-note repair probe:

- Issue #461 result: boundary `stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe`
- source valid / strict / grammar gate sample count: `1` / `1` / `3`
- repair valid / strict / grammar gate sample count: `2` / `2` / `3`
- source / repair long-note failure count: `2` / `0`
- long-note failure delta: `2`
- valid / strict sample delta: `1` / `1`
- onset / sustained coverage delta: `0.020833333333333343` / `-0.2708333333333333`
- coverage regression observed: `true`
- remaining failure reason: `dead-air ratio too high: 0.800 >= 0.800` `1`
- raw generation quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint duration long-note remaining blocker decision이다.

현재 generic base scale checkpoint duration long-note remaining blocker decision:

- Issue #463 result: boundary `stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision`
- decision: `select_sustained_coverage_dead_air_repair_probe`
- selected target: `sustained_coverage_dead_air_repair`
- remaining blocker: `sustained_coverage_dead_air`
- valid / strict / grammar gate sample count: `2` / `2` / `3`
- long-note failure count: `0`
- dead-air failure count: `1`
- coverage regression observed: `true`
- onset / sustained coverage delta: `0.020833333333333343` / `-0.2708333333333333`
- audio review selected: `false`
- musical quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint sustained coverage dead-air repair probe다.

현재 generic base scale checkpoint sustained coverage dead-air repair probe:

- Issue #465 result: boundary `stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe`
- constrained note groups per bar: `8`
- baseline valid / strict / grammar gate sample count: `2` / `2` / `3`
- repair valid / strict / grammar gate sample count: `3` / `3` / `3`
- baseline dead-air / long-note failure count: `1` / `0`
- repair dead-air / long-note failure count: `0` / `0`
- dead-air failure delta: `1`
- valid / strict sample delta: `1` / `1`
- onset / sustained coverage delta: `0.19791666666666669` / `0.2708333333333333`
- max longest sustained empty run steps: `8 -> 4`
- remaining failure reason: none
- raw generation quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint objective gate consolidation이다.

현재 generic base scale checkpoint objective gate consolidation:

- Issue #467 result: boundary `stage_b_generic_base_scale_checkpoint_objective_gate_consolidation`
- decision: `select_objective_gate_repeatability_sweep`
- selected target: `objective_gate_repeatability_sweep`
- objective gate support: `true`
- single seed set only: `true`
- valid / strict / grammar gate sample count: `3` / `3` / `3`
- dead-air / long-note failure count: `0` / `0`
- avg onset / sustained coverage: `0.3854166666666667` / `0.6354166666666666`
- repeatability / musical quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint objective gate repeatability sweep이다.

현재 generic base scale checkpoint objective gate repeatability sweep:

- Issue #469 result: boundary `stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep`
- objective gate repeatability target qualified: `true`
- repeatability claimed: `true`
- seeds: `[44, 52, 60]`
- seed count: `3`
- sample count: `9`
- valid / strict / grammar gate sample count: `9` / `9` / `9`
- avg onset / sustained coverage: `0.4236111111111111` / `0.6805555555555556`
- max longest sustained empty run steps: `4`
- failure reasons: none
- raw generation quality / broad model quality / Brad style adaptation claim: `false` / `false` / `false`
- 다음 작업은 generic base scale checkpoint repeatability consolidation이다.

### Phase 5. Brad Style Adaptation

목표:

- generic jazz base 위에 Brad subset adaptation을 검토한다.

조건:

- generic base가 먼저 valid solo-line MIDI를 만들 수 있어야 한다.
- Brad 72 files 전체를 scratch로 학습하는 방향은 우선순위가 낮다.

후보:

- adapter fine-tuning
- LoRA on real pretrained/base checkpoint
- retrieval/motif memory
- style token conditioning

### Phase 6. Product/Serving MVP

목표:

- 모델 core가 reviewable output을 만들 때만 backend/API로 확장한다.

후순위 작업:

- FastAPI inference server
- request schema
- MIDI download path
- job status
- Spring Boot backend
- DAW/live integration

지금은 하지 않는다.

## 7. 레퍼런스 기준으로 맞는가

현재 방향은 레퍼런스와 대체로 맞다.

맞는 부분:

- Music Transformer 계열 symbolic sequence model을 사용한다.
- REMI/Jazz Transformer처럼 bar/position/chord/duration을 명시한다.
- full-song sequence 대신 phrase/window dataset으로 줄인다.
- tiny-overfit과 decode/review gate를 먼저 통과시키려 한다.
- 작은 Brad dataset만으로 style을 scratch 학습하지 않으려 한다.

아직 부족한 부분:

- MidiTok 같은 검증된 tokenizer library를 직접 사용하지 않았다.
- pretrained symbolic MIDI model을 아직 평가하지 않았다.
- Compound Word/Octuple 같은 grouped representation은 아직 구현하지 않았다.
- chord inference/lead-sheet alignment는 아직 약하다.
- musical listening review loop가 자동화되어 있지 않다.

판단:

> 지금은 "논문 구현체 복제"가 아니라 "논문들이 말하는 실패 방지 순서에 맞춘 local engineering path"다.

## 8. 앞으로 하지 말아야 할 것

다음은 금지하거나 뒤로 미룬다.

- one passing MIDI를 보고 broad training으로 바로 넘어가기
- postprocess를 더 세게 해서 모델 성공처럼 보이게 만들기
- Spring Boot/API/UI를 다시 MVP 중심으로 가져오기
- Brad-only tiny dataset으로 "style model"이라고 주장하기
- `valid .mid file exists`를 성공으로 처리하기
- exact artist clone처럼 공개적으로 표현하기
- SOTA 모델 이름만 붙이고 evaluation 없이 진행하기

## 9. 다음 바로 할 일

완료된 바로 전 작업:

```text
Stage B margin-recovered phrase/vocabulary duration coverage fill focused listening fill
```

결과:

- selected candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- docs: `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_DURATION_COVERAGE_FILL_FOCUSED_LISTENING_FILL_2026-05-29.md`
- candidate count: `1`
- prior decision: `keep_for_focused_listening`
- listening decision: `keep`
- reviewed count: `1`
- pending count: `0`
- review risks: `[]`
- timing: `acceptable`
- chord fit: `strong`
- phrase continuation: `acceptable`
- landing: `strong`
- jazz vocabulary: `acceptable`
- note count: `18`
- unique pitch count: `15`
- range: `D#4-G#5`
- phrase span: `7.000` beats
- max active notes: `1`
- dead-air ratio: `0.2941`
- onset coverage: `0.5625`
- sustained coverage: `0.6250`
- adjacent pitch repeats: `0`
- duplicated 3-note pitch-class chunks: `0`
- max interval: `7`
- final note: `F4` over `Fm7`, chord tone

판단:

- MIDI/context evidence fill 기준 keep.
- source coverage metric 부재 시 solo MIDI 기반 coverage를 산출하도록 보정.
- adjacent repeat, wide interval blocker repair 유지.
- human/audio listening proof는 아직 아니다.
- claim boundary: `postprocess_duration_coverage_fill_candidate`.
- broad trained-model quality, human listening preference, Brad style adaptation은 아직 미검증이다.

후속:

- Issue #322 keep consolidation 완료.
- Issue #324 human/audio boundary 완료.
- broad training은 focused context/listening boundary를 먼저 본 뒤 결정한다.

## 9.1 Stage B margin-recovered phrase/vocabulary duration coverage fill keep consolidation

Issue #322는 Issue #320의 `keep` 결과를 claim boundary 기준으로 정리한 작업이다.

결과:

- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- decision: `keep`
- evidence boundary: `single_postprocess_candidate_keep_support`
- postprocess claim boundary: `postprocess_duration_coverage_fill_candidate`
- variant count: `4`
- qualified variant count: `2`
- fill additions: `6`
- dead-air ratio: `0.5714 -> 0.2941`
- onset coverage: `0.5625`
- sustained coverage: `0.6250`
- note count: `18`
- unique pitch count: `15`
- final note: `F4` over `Fm7`, chord tone

판단:

- MIDI/context evidence keep은 확인했다.
- adjacent repeat blocker와 wide interval blocker는 repair 상태다.
- single postprocess candidate support이므로 broad repeatability는 아직 아니다.
- human/audio preference, broad trained-model quality, Brad style adaptation은 아직 미검증이다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-keep-consolidation`

다음 작업:

- `Stage B margin-recovered phrase/vocabulary duration coverage fill external human/audio review boundary`

## 9.2 Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio boundary

Issue #324는 duration/coverage fill keep 후보의 human/audio review boundary를 정의한 작업이다.

결과:

- review item count: `2`
- human/audio status: `pending`
- boundary: `pending_human_audio_review_source_vs_fill_distinct_midi_content`
- preference claimed: `false`
- note sequence match: `false`
- metric summary match: `false`
- fill additions: `6`
- dead-air delta: `0.2773`
- source note signature count: `15`
- selected note signature count: `18`

판단:

- source constrained partial과 duration fill keep 후보는 MIDI content가 다르다.
- human/audio preference는 아직 입력되지 않았다.
- audio render quality, broad trained-model quality, Brad style adaptation은 아직 미검증이다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-boundary`

후속:

- Issue #326 human/audio review input guard 완료.
- Issue #328 audio review package 완료.
- Issue #330 MIDI evidence review 완료.
- Issue #332 MIDI evidence consolidation 완료.

## 9.3 Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio review input guard

Issue #326은 duration/coverage fill human/audio review fill에서 review input 없이 preference가 채워지는 것을 막는 작업이다.

결과:

- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- review input present: `false`
- fill status: `pending_review_input`
- human/audio status: `pending`
- preference: `pending`
- preference claimed: `false`
- audio render used: `false`

판단:

- review input absent 상태에서 preference claim 차단.
- pending status 유지.
- human/audio preference와 audio rendered quality는 아직 미검증이다.
- review input present 경로는 reviewer, audio_render_used, preference schema를 검증한다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-review-input-guard`

다음 작업:

- `Stage B margin-recovered phrase/vocabulary duration coverage fill external human/audio review boundary`

## 9.6 Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence consolidation

Issue #332는 Issue #330 MIDI evidence review 결과의 claim boundary를 정리한 작업이다.

결과:

- boundary: `midi_evidence_preference_support`
- preference: `duration_coverage_fill_keep`
- source score: `91.857`
- fill score: `171.588`
- score delta fill-source: `79.7311`
- dead-air delta fill-source: `-0.2773`
- focused note count delta: `+6`
- focused unique pitch count delta: `+6`
- max simultaneous notes delta: `-1`
- human/audio preference claimed: `false`

판단:

- MIDI metric preference for duration/coverage fill candidate 확인.
- source partial 대비 dead-air 감소, focused note count 증가, focused unique pitch count 증가.
- human/audio preference와 audio rendered quality는 아직 미검증이다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-consolidation`

다음 작업:

- `Stage B margin-recovered phrase/vocabulary duration coverage fill external human/audio review boundary`

## 9.5 Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review

Issue #330은 source constrained partial과 duration/coverage fill 후보를 MIDI evidence 기준으로 비교한 작업이다.

결과:

- review basis: `midi_metric_and_note_structure`
- MIDI evidence preference: `duration_coverage_fill_keep`
- source score: `91.857`
- fill score: `171.588`
- score delta fill-source: `79.7311`
- dead-air delta fill-source: `-0.2773`
- focused note count delta: `+6`
- focused unique pitch count delta: `+6`
- max simultaneous notes delta: `-1`
- human/audio preference claimed: `false`
- audio render used: `false`

판단:

- MIDI evidence 기준 fill 후보 우세.
- fill 후보는 source 대비 dead-air 감소, focused note count 증가, focused unique pitch 증가.
- adjacent repeat, duplicated 3-note pitch-class chunk, max interval guardrail 유지.
- audio render와 human/audio preference는 아직 미검증이다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-review`

다음 작업:

- `Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review consolidation`

## 9.4 Stage B margin-recovered phrase/vocabulary duration coverage fill audio review package

Issue #328은 duration/coverage fill 후보의 외부 review input 전 package를 만든 작업이다.

결과:

- review item count: `2`
- package status: `ready_for_external_review_input`
- audio render status: `not_rendered_by_harness`
- preference claimed: `false`
- required file count: `3`
- source MIDI exists: `true`
- selected MIDI exists: `true`
- selected context MIDI exists: `true`
- source MIDI sha256 prefix: `8429ccb789ba`
- selected MIDI sha256 prefix: `b517b822a919`

판단:

- external review input 전 package 준비 완료.
- source/fill MIDI와 selected context MIDI 파일 존재 및 checksum 확인.
- harness audio render는 수행하지 않았다.
- preference claim은 여전히 없다.

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-audio-review-package`

다음 작업:

- `Stage B margin-recovered phrase/vocabulary duration coverage fill external review input fill`

## 10. 한 문장 요약

이 프로젝트의 현재 핵심은 다음이다.

> Brad-style jazz MIDI model을 바로 만드는 것이 아니라, reviewable jazz solo-line MIDI를 만들 수 있는 symbolic representation, dataset window, generation, decoding, and evaluation pipeline을 먼저 증명하는 것이다.
