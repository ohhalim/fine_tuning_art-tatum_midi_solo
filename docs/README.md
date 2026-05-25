# Docs Index

작성일: 2026-05-20

이 디렉터리는 현재 jazz piano MIDI fine-tuning probe를 진행하기 위한 기준 문서만 전면에 둔다. 백엔드/API/ERD/제품 MVP 문서는 `docs/archive/`로 이동했다.

## Active Docs

- `CODEX_REMOTE_HANDOFF_2026-05-23.md`
  - Codex 앱/원격 세션이 최신 상태, 목표, 다음 작업을 바로 이해하기 위한 handoff 문서.
- `CURRENT_STATUS_AND_PLAN.md`
  - 현재 브랜치 상태, 결정 사항, 다음 실행 순서.
- `BRAD_MEHLDAU_FINETUNING_PLAN.md`
  - Brad Mehldau MIDI dataset audit, training probe order, acceptance criteria.
- `DATASET_STRATEGY.md`
  - 전체 jazz piano corpus audit, generic jazz pianist base, Brad style adaptation 전략.
- `STAGE_A_TOKEN_FORMAT.md`
  - `control_v1` token sequence, legacy format, checkpoint vocab migration 규칙.
- `STAGE_A_TRAINING_MODES.md`
  - full checkpoint/from-scratch training, adapter training, LoRA-only mode 경계.
- `STAGE_A_TINY_OVERFIT.md`
  - tiny-overfit smoke 실행 방법과 통과/실패 판단 기준.
- `STAGE_A_CODE_REVIEW_2026-05-18.md`
  - sustain block/chord block MIDI가 나온 원인과 다음 수정 방향.
- `STAGE_A_BRAD_PROBE2_2026-05-18.md`
  - Brad 2-file `control_v1` training/generation probe 결과와 Stage B 전환 판단.
- `STAGE_B_TOKENIZATION_SPEC.md`
  - duration-explicit, bar-position-aware Stage B tokenization contract.
- `STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`
  - `prepare_role_dataset.py --sequence_format stage_b_v1` 연결 결과와 Brad 2-file dry run.
- `STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`
  - Stage B 2-bar phrase/window dataset extraction and Brad 2-file dry run.
- `STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`
  - Stage B phrase windows가 model vocab/training path에 연결되는지 확인한 tiny-overfit smoke.
- `STAGE_B_GENERATION_PROBE_2026-05-19.md`
  - Stage B token generation, MIDI decode, and review-gate probe result.
- `STAGE_B_CONSTRAINED_TINY_OVERFIT_2026-05-19.md`
  - Stage B constrained note-group generation and grammar-gate result.
- `STAGE_B_OVERLAP_GATE_2026-05-19.md`
  - Stage B constrained output overlap/dedup postprocess and first local review-gate pass.
- `STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`
  - Stage B multi-sample constrained probe, pass-rate reporting, and sampling collapse negative control.
- `STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`
  - Stage B collapse diagnostics and `top_k` sampling sweep result.
- `STAGE_B_STRICT_COLLAPSE_GATE_2026-05-20.md`
  - Stage B strict collapse-aware review gate and basic-vs-strict sweep result.
- `STAGE_B_2FILE_BRAD_PROBE_2026-05-20.md`
  - Stage B Brad 2-file generation probe and dead-air/temporal coverage failure result.
- `STAGE_B_TEMPORAL_COVERAGE_DIAGNOSTICS_2026-05-20.md`
  - Stage B token-level temporal coverage diagnostics and next coverage-aware generation boundary.
- `STAGE_B_COVERAGE_AWARE_GENERATION_2026-05-20.md`
  - Stage B coverage-aware constrained `POSITION` generation result and dead-air gate comparison.
- `STAGE_B_COVERAGE_AB_SWEEP_2026-05-20.md`
  - Stage B plain-vs-coverage A/B sweep across note-group density settings.
- `STAGE_B_CANDIDATE_RANKING_2026-05-20.md`
  - Stage B generated MIDI candidate ranking report for listening/review priority.
- `STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`
  - Stage B ranking이 low chord-tone/repeated pitch/mechanical pattern MIDI를 좋은 후보로 올리지 않도록 고친 결과.
- `STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`
  - Stage B constrained generation에서 chord-aware pitch 후보군을 적용한 결과와 reviewable candidate 회복.
- `STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`
  - Stage B `coverage_chord` top candidates를 manual listening review용 manifest/markdown으로 export하는 도구.
- `STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`
  - 2-bar 후보가 너무 짧다는 piano-roll review를 반영해 4-bar `coverage_chord` phrase 후보를 생성/export하는 probe.
- `STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`
  - 4-bar 후보의 repeated-pitch risk가 adjacent collapse인지 제한된 pitch-set 재사용인지 구분하는 contour diagnostics.
- `STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
  - "근음을 계속 치는 느낌"을 root-tone ratio와 tension ratio로 분리해 진단하는 문서.
- `STAGE_B_PITCH_MODE_COMPARE_2026-05-21.md`
  - `tones`와 `tones_tensions`를 같은 Stage B 4-bar 조건에서 비교한 결과와 다음 phrase-shape control 판단 기준.
- `STAGE_B_8BAR_APPROACH_PHRASE_2026-05-21.md`
  - 8-bar `tones`/`tones_tensions`/`approach_tensions` 비교와 beginner-like melodic exercise 상태 진단.
- `STAGE_B_SWING_MOTIF_PHRASE_2026-05-21.md`
  - 8-bar `approach_tensions` 위에 swing/motif position-duration grammar를 얹어 기계적인 rhythm-grid 반복을 줄인 결과.
- `STAGE_B_REFERENCE_PHRASE_STATS_2026-05-21.md`
  - 실제 Stage B jazz MIDI phrase window 통계를 만들어 generated rhythm/motif 후보와 비교한 결과.
- `STAGE_B_MOTIF_TEMPLATE_EXTRACTION_2026-05-21.md`
  - 실제 Stage B phrase window에서 rhythm/contour/full motif templates를 추출해 다음 data-derived generation constraint로 넘기기 위한 결과.
- `STAGE_B_DATA_MOTIF_GENERATION_2026-05-21.md`
  - 추출된 motif catalog를 8-bar generation baseline에 연결해 hand-written swing grammar와 비교한 결과.
- `STAGE_B_DATA_MOTIF_REVIEW_EXPORT_2026-05-21.md`
  - data-derived motif baseline과 hand-written swing baseline의 MIDI 후보를 named review package로 export한 결과.
- `STAGE_B_REVIEW_CONTEXT_GRID_2026-05-22.md`
  - solo-only 리뷰의 한계를 반영해 chord/bass context MIDI와 straight-grid timing reference를 추가한 결과.
- `STAGE_B_GUIDE_TONE_CADENCE_2026-05-22.md`
  - straight-grid timing 위에서 scale/chromatic 나열을 줄이기 위한 guide-tone/cadence 후보를 추가한 결과.
- `STAGE_B_DATA_GUIDE_HYBRID_2026-05-22.md`
  - data-derived motif rhythm과 guide-tone/cadence pitch vocabulary를 결합한 hybrid 후보를 추가한 결과.
- `STAGE_B_REFERENCE_PITCH_ROLE_STATS_2026-05-22.md`
  - reference phrase window에서 pitch-role landing 통계를 시도했고 chord annotation coverage가 없다는 blocker를 확인한 결과.
- `STAGE_B_CHORD_COVERAGE_AUDIT_2026-05-22.md`
  - role metadata, raw sidecar, MIDI text event를 훑어 현재 dataset에 usable chord progression annotation이 없음을 확인한 결과.
- `STAGE_B_CHORD_LABELED_EVAL_2026-05-22.md`
  - known chord labels가 있을 때 pitch-role summary를 계산할 수 있는 tiny evaluation contract와 manifest format.
- `STAGE_B_GENERATED_CHORD_EVAL_2026-05-22.md`
  - generated candidate report의 known chord progression metadata를 chord-labeled evaluator에 연결하는 bridge.
- `STAGE_B_DATA_GUIDE_GENERATED_CHORD_EVAL_2026-05-22.md`
  - data-guide hybrid review package에 generated chord eval bridge를 적용해 `data_motif`와 `data_motif_guide_tones` pitch-role profile을 비교한 결과.
- `STAGE_B_REVIEW_MARKDOWN_CHORD_EVAL_2026-05-22.md`
  - review candidates markdown과 generated chord eval summary를 결합해 청취 리뷰용 한 파일로 만든 결과.
- `STAGE_B_LISTENING_REVIEW_NOTES_2026-05-22.md`
  - 청취 리뷰 판단을 후보별 enum/notes schema로 기록하기 위한 template generator 결과.
- `STAGE_B_LISTENING_REVIEW_AGGREGATE_2026-05-22.md`
  - 사람이 채운 listening review notes를 다음 generation rule 후보로 집계하는 도구 결과.
- `STAGE_B_FULL_REVIEW_MANIFEST_NOTES_2026-05-22.md`
  - 전체 review manifest 후보 15개를 파일 경로와 함께 listening review notes로 변환한 결과.
- `STAGE_B_OBJECTIVE_MIDI_NOTE_REVIEW_2026-05-22.md`
  - generated review MIDI를 직접 읽어 overlap, grid, scalar/chromatic, duration collapse를 진단한 결과.
- `STAGE_B_OBJECTIVE_FLAGS_REVIEW_FLOW_2026-05-22.md`
  - objective MIDI flags를 listening review notes와 aggregate priority에 연결한 결과.
- `STAGE_B_OVERLAP_FREE_REVIEW_EXPORT_2026-05-22.md`
  - overlap-free solo-line review MIDI variant를 export하고 objective overlap/polyphonic flag를 제거한 결과.
- `STAGE_B_DURATION_VARIATION_REVIEW_2026-05-22.md`
  - varied-duration baseline을 추가해 review MIDI의 duration collapse flag를 제거한 결과.
- `STAGE_B_PHRASE_CADENCE_REVIEW_2026-05-22.md`
  - phrase/cadence baseline을 추가해 scalar/chromatic objective flags를 줄인 결과.
- `STAGE_B_PHRASE_NATURALNESS_OBJECTIVES_2026-05-22.md`
  - 큰 도약 뒤 회복 움직임이 없는 phrase naturalness risk를 objective flag로 추가한 결과.
- `STAGE_B_PHRASE_RECOVERY_REVIEW_2026-05-22.md`
  - 큰 도약 뒤 반대 방향 small recovery를 넣는 phrase recovery baseline 결과.
- `STAGE_B_DATA_MOTIF_PHRASE_RECOVERY_2026-05-22.md`
  - data-derived motif rhythm과 phrase recovery pitch grammar를 결합한 결과.
- `STAGE_B_CLEAN_REVIEW_PACKAGE_2026-05-23.md`
  - objective-clean `data_motif_phrase_recovery` 후보만 골라 context MIDI와 함께 listening review package로 묶은 결과.
- `STAGE_B_CLEAN_CONTEXT_DIAGNOSTICS_2026-05-23.md`
  - clean context MIDI 후보 3개를 note-level로 다시 읽어 coverage/timing/pitch-reuse 진단을 남긴 결과.
- `STAGE_B_CLEAN_LISTENING_REVIEW_NOTES_2026-05-23.md`
  - objective-clean context 후보 3개를 같은 schema로 review할 수 있는 notes template 결과.
- `STAGE_B_CLEAN_MIDI_PROXY_REVIEW_2026-05-24.md`
  - clean 후보 3개를 MIDI note/context track 기준으로 proxy review하고 다음 contour/cadence landing repair 방향을 정리한 결과.
- `STAGE_B_CONTOUR_LANDING_REPAIR_2026-05-25.md`
  - data-derived rhythm/contour 후보에 cadence landing repair를 추가해 final landing과 register contour objective risk를 줄인 결과.
- `STAGE_B_CONTOUR_REPAIR_MIDI_PROXY_REVIEW_2026-05-25.md`
  - contour/landing repair 후보와 phrase recovery baseline 후보를 MIDI-note/context 기준으로 proxy review하고 다음 rhythm/phrase vocabulary 방향을 정리한 결과.
- `STAGE_B_RHYTHM_PHRASE_VARIATION_2026-05-25.md`
  - contour/landing repair를 유지하면서 duration/IOI template variation과 register floor를 추가한 결과.
- `STAGE_B_RHYTHM_PHRASE_VARIATION_MIDI_PROXY_REVIEW_2026-05-25.md`
  - rhythm/phrase variation 후보를 MIDI-note/context 기준으로 proxy review하고 exact duplicate sample-diversity 문제를 확인한 결과.
- `STAGE_B_RHYTHM_VARIATION_SAMPLE_DIVERSITY_2026-05-25.md`
  - variation 후보 rank 1-3이 exact duplicate가 되지 않도록 seed-driven sequence variation과 duplicate detection을 추가한 결과.
- `STAGE_B_SAMPLE_DIVERSE_RHYTHM_PROXY_REVIEW_2026-05-25.md`
  - sample-diverse rhythm variation 후보를 MIDI-note/context 기준으로 proxy review하고 timing-grid repetition repair 방향을 정리한 결과.
- `STAGE_B_RHYTHM_VARIATION_TIMING_GRID_REPAIR_2026-05-25.md`
  - rhythm variation 후보의 dominant IOI repetition을 줄이고 max interval/objective leap risk를 정리한 결과.
- `STAGE_B_TIMING_GRID_REPAIRED_PROXY_REVIEW_2026-05-25.md`
  - timing-grid repaired rhythm 후보를 MIDI-note/context 기준으로 proxy review하고 phrase-vocabulary diversity repair 방향을 정리한 결과.
- `STAGE_B_RHYTHM_VARIATION_PHRASE_VOCAB_REPAIR_2026-05-25.md`
  - timing-grid guardrail을 유지하면서 rhythm variation 후보의 bar-position/IOI/phrase vocabulary를 넓힌 결과.
- `STAGE_B_PHRASE_VOCAB_REPAIRED_PROXY_REVIEW_2026-05-25.md`
  - phrase-vocabulary repaired rhythm 후보를 MIDI-note/context 기준으로 proxy review하고 phrase-shape/tension repair 방향을 정리한 결과.
- `STAGE_B_RHYTHM_VARIATION_PHRASE_SHAPE_TENSION_REPAIR_2026-05-25.md`
  - rhythm/position guardrail을 유지하면서 phrase target register와 tension pitch-class 우선순위를 보강한 결과.
- `STAGE_B_PHRASE_SHAPE_TENSION_PROXY_REVIEW_2026-05-25.md`
  - phrase-shape/tension repaired rhythm 후보를 MIDI-note/context 기준으로 proxy review하고 첫 proxy keep 후보를 정리한 결과.
- `STAGE_B_PROXY_KEEP_FOCUSED_REVIEW_PACKAGE_2026-05-25.md`
  - 첫 proxy keep 후보만 solo/context MIDI와 objective note summary로 묶은 focused review package 결과.
- `STAGE_B_PROXY_KEEP_FOCUSED_CONTEXT_DECISION_2026-05-25.md`
  - focused package의 단일 proxy keep 후보를 context MIDI note 기준으로 다시 판단하고 register/cadence blocker를 정리한 결과.
- `REFERENCES.md`
  - 2024-2026 symbolic MIDI 연구까지 포함한 fine-tuning/tokenization reference map과 구현 판단 기준.
- `INFERENCE_MODEL_SPEC.md`
  - request-conditioned generation, fallback, metrics, post-processing contract.
- `QA_ACCEPTANCE_PLAN.md`
  - MIDI output을 리뷰 가능한 샘플로 인정하기 위한 gate.
- `RUNPOD_GUIDE.md`
  - GPU training 환경 참고.

## Archived Docs

`docs/archive/`에는 현재 브랜치 범위가 아닌 문서가 있다.

- Spring Boot/API/ERD/backend MVP 문서
- 한 달 포트폴리오 로드맵
- realtime/DAW/plugin/product-planning 문서
- 오래된 리뷰와 외부 리서치 정리

이 문서들은 삭제하지 않고 보관한다. 현재 모델 fine-tuning이 reviewable MIDI를 만들기 전까지는 active plan으로 취급하지 않는다.

## Current Execution Order

1. Full jazz piano corpus audit 결과를 기준으로 generic base 후보 데이터를 확인한다.
2. Brad Mehldau subset은 style adaptation과 holdout evaluation 용도로 분리한다.
3. `max_files=2` Brad `control_v1` probe 결과를 기준으로 Stage A 한계를 문서화한다.
4. broad training 전에 duration-explicit Stage B tokenization과 phrase/window dataset을 설계한다.
5. Stage B phrase/window tiny-overfit, constrained grammar probe, overlap gate, multi-sample probe, collapse sweep, strict collapse gate, 2-file generation probe, temporal coverage probe, coverage-aware constrained probe, coverage-aware A/B sweep, candidate ranking, harmonic/repetition gate, chord-aware pitch probe, candidate review export, longer 4-bar phrase probe, phrase contour diagnostics, root-bias diagnostics, pitch-mode comparison, 8-bar approach phrase probe, swing/motif phrase grammar probe, real phrase reference statistics, motif template extraction, data-derived motif baseline generation, data motif review export, review context/grid export, reference pitch-role stats, chord coverage audit, chord-labeled eval contract, generated chord eval bridge, data-guide generated chord eval, review markdown chord eval summary, listening review notes schema, filled listening review aggregate, full review manifest notes, objective MIDI note review, objective flags review flow, overlap-free review export, duration variation review, phrase/cadence review, phrase naturalness objective review, phrase recovery review, data motif phrase recovery review, objective clean review package, clean context diagnostics, clean listening review notes, clean MIDI-note proxy review, contour/cadence landing repair probe, contour repair MIDI-note proxy review, rhythm/phrase vocabulary variation probe, rhythm/phrase variation MIDI-note proxy review, rhythm/phrase variation sample diversity repair, sample-diverse rhythm variation MIDI-note proxy review, rhythm variation timing-grid repetition repair, timing-grid repaired rhythm MIDI-note proxy review, rhythm variation phrase-vocabulary diversity repair, phrase-vocabulary repaired rhythm MIDI-note proxy review, rhythm variation phrase-shape tension repair, and phrase-shape tension repaired MIDI-note proxy review를 통과한 뒤 generic jazz base 학습 여부를 다시 결정한다.

핵심 원칙:

> 모델이 valid solo-line MIDI를 만들기 전에는 백엔드, UI, realtime 통합을 확장하지 않는다.
