# Jazz Piano MIDI Solo Generator MVP

입력 MIDI/chord context를 받아 짧은 jazz piano solo-line MIDI 후보를 만들고, 후보별 WAV와 objective review package까지 생성하는 symbolic MIDI model-core MVP.

## 현재 상태

- 생성 방식: Music Transformer 계열 symbolic checkpoint + constrained decoding
- 출력 단위: 2-4마디 solo-line MIDI 후보
- 최신 review package: MIDI `8`, WAV `8`
- objective rubric: pass/fail `6 / 2`
- 남은 major label: `low_tension_color=2`
- 남은 watch label: `dead_air_watch=3`
- tension 추가 repair 가능성: current guard 기준 `false`
- validated listening input: `false`
- human/audio preference claim: `false`
- musical quality claim: `false`

## 구현 범위

- MIDI dataset tokenization
- Music Transformer 계열 symbolic checkpoint 학습 경로
- checkpoint logits 기반 candidate generation
- chord/rhythm 범위 기반 constrained decoding
- MIDI 문법 gate
- objective metric 계산
- 후보 ranking 및 MIDI export
- FluidSynth 기반 WAV render
- review package 생성
- listening input guard
- residual risk 문서화

## 최근 개선 흐름

- broader repaired sampling repeatability: strict `40 / 40`, grammar-valid `40 / 40`
- final handoff audit: MIDI/WAV `8 / 8`, missing `0`, checksum mismatch `0`
- objective quality rubric baseline: pass/fail `3 / 5`
- dead-air density repair: dead-air avg `0.6583 -> 0.5806`, high count `2 -> 0`
- phrase direction repair: direction avg `0.4912 -> 0.6016`, direction low count `4 -> 0`
- tension color repair: tension avg `0.1979 -> 0.2431`, low count `4 -> 2`
- residual tension feasibility decision: tension repeat feasible `false`
- rhythm syncopation repair: syncopation avg `0.7951 -> 0.8160`, low count `1 -> 0`
- residual-aware final review package: candidate `8`, MIDI/WAV `8 / 8`, proxy pass/fail `6 / 2`
- residual-aware listening input guard: preference fill `false`, listening review completed `false`
- residual-aware MVP handoff freeze: local artifacts verified `true`, checksum mismatch `0`
- residual-aware listening review pending: pending candidate `8`, quality claim `false`

## 최신 산출물

- final review package: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.md`
- review input template: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_review_input_template.json`
- MIDI 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/midi/`
- WAV 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/audio/`
- final review doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- listening input guard doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- MVP handoff freeze doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- listening review pending doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`

## 재현 명령

```bash
.venv/bin/python scripts/build_music_transformer_solo_yield_residual_aware_final_review_package.py \
  --run_id issue_1388_residual_aware_final_review_package \
  --source_package outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/rhythm_syncopation_balance_repair_package.json \
  --rubric_report outputs/music_transformer_finetune_mvp/solo_yield_objective_quality_rubric/issue_1386_rhythm_syncopation_repair_rubric_review/objective_quality_rubric_baseline.json \
  --feasibility_decision outputs/music_transformer_finetune_mvp/solo_yield_residual_tension_decision/issue_1382_residual_tension_feasibility_decision/residual_tension_target_decision.json \
  --min_candidate_count 8 \
  --require_residual_context \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/guard_music_transformer_solo_yield_residual_aware_listening_input.py \
  --run_id issue_1390_residual_aware_listening_input_guard \
  --source_package outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_status_sync \
  --require_pending_input \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/freeze_music_transformer_solo_yield_residual_aware_mvp_handoff.py \
  --run_id issue_1396_residual_aware_mvp_handoff_freeze \
  --final_review_package outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.json \
  --input_guard_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_input_guard/issue_1390_residual_aware_listening_input_guard/residual_aware_listening_input_guard.json \
  --status_audit_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_status_audit/issue_1394_residual_aware_status_audit/residual_aware_status_audit.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_listening_review_pending \
  --require_local_artifacts_verified \
  --require_pending_input \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_pending.py \
  --run_id issue_1398_residual_aware_listening_review_pending \
  --handoff_freeze_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_mvp_handoff_freeze/issue_1396_residual_aware_mvp_handoff_freeze/residual_aware_mvp_handoff_freeze.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_completion_audit \
  --require_pending_review \
  --require_no_quality_claim
```

## 아닌 것

- 완성형 jazz solo quality claim 아님
- 특정 pianist style adaptation 완료 아님
- production-ready improviser 아님
- human listening preference 검증 완료 아님
- raw audio generation 프로젝트 아님

## 주요 파일

- `scripts/build_music_transformer_solo_yield_rhythm_syncopation_balance_repair_package.py`
- `scripts/build_music_transformer_solo_yield_residual_aware_final_review_package.py`
- `scripts/guard_music_transformer_solo_yield_residual_aware_listening_input.py`
- `scripts/freeze_music_transformer_solo_yield_residual_aware_mvp_handoff.py`
- `scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_pending.py`
- `scripts/build_music_transformer_solo_yield_objective_quality_rubric_baseline.py`
- `scripts/decide_music_transformer_solo_yield_residual_tension_target.py`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`
