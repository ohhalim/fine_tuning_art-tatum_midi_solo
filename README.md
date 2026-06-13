# Jazz Piano MIDI Solo Generator MVP

입력 MIDI/chord context를 받아 짧은 jazz piano solo-line MIDI 후보를 만들고, 후보별 WAV와 objective review package까지 생성하는 symbolic MIDI model-core MVP.

## 현재 상태

- 생성 방식: Music Transformer 계열 symbolic checkpoint + constrained decoding
- 출력 단위: 2-8마디 solo-line MIDI 후보
- 최신 대표 review package: strict-listen top4 bar-repair, MIDI `4`, WAV `4`
- 최신 objective gate: max gate penalty `0.0000`
- 최신 chord/rhythm 지표: strong-beat chord-tone `1.0000`, unresolved offbeat `0.0156`, interval repeat `0.0000`, bar similarity `0.6012`
- 기준 best-of review package: MIDI `16`, WAV `16`
- residual-aware objective rubric: pass/fail `6 / 2`
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
- residual-aware completion audit: technical MVP complete `true`, local review ready `true`
- residual-aware final status sync: synced `true`, next `listening_review_input_wait`
- residual-aware listening review input wait: quality claim blocked `true`
- bebop language best-of strict consonance listen-first package: source package `107`, pool `1519`, selection pool `64`, selected `16`, score `0.1905`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4219`, offbeat resolution `0.9256`, unresolved offbeat non-chord `0.0313`, altered offbeat `0.1797`, two-note cycle `0.0092`, interval trigram repeat `0.0369`, max gate penalty `0.0000`, listen-first mode `consonance`
- bebop language strict-listen top4 bar-repair package: source package `119`, pool `1711`, selection pool `16`, selected `4`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3984`, offbeat resolution `0.9615`, unresolved offbeat non-chord `0.0156`, altered offbeat `0.1875`, two-note cycle `0.0000`, interval trigram repeat `0.0000`, bar pitch-class similarity `0.6012`, max gate penalty `0.0000`, major ii-V turnaround bar similarity `0.8000 -> 0.5000`
- bebop language best-of with-sweeps package: source package `91`, pool `1263`, selected `16`, score `0.2143`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4277`, offbeat resolution `0.9177`, unresolved offbeat non-chord `0.0352`, altered offbeat `0.1836`, two-note cycle `0.0082`, max gate penalty `0.0639`
- bebop language best-of balanced package: source package `33`, pool `335`, selected `16`, score `0.2728`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4414`, offbeat resolution `0.8983`, unresolved offbeat non-chord `0.0449`, altered offbeat `0.1602`, two-note cycle `0.0092`, max-per-case `4`
- bebop language altered-color balanced package: generated `4000`, selected `16`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4512`, offbeat resolution `0.8914`, unresolved offbeat non-chord `0.0488`, unique pitch avg `14.8125`, 3rd/4th motion `0.4831`, large leap `0.0863`, altered offbeat `0.1445`, bar pitch-class similarity `0.7027`, half-repeat `0.0000`
- bebop language bar-similarity package: generated `1440`, selected `16`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4746`, offbeat resolution `0.8891`, unresolved offbeat non-chord `0.0527`, bar pitch-class similarity `0.7280`, half-repeat `0.0000`
- bebop language data-contour sweep: config `18`, best `config_01`, strong-beat chord-tone `1.0000`, offbeat resolution `0.8598`, unresolved offbeat non-chord `0.0625`, unique pitch avg `13.9375`, 3rd/4th motion `0.4931`, bar pitch-class similarity `0.7449`, half-repeat `0.0000`

## 최신 산출물

- best-of strict consonance 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/listen_first_by_progression/`
- best-of strict consonance 전체 WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/audio_with_context/`
- best-of strict consonance package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/bebop_language_best_of_package.md`
- best-of strict consonance note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_v8_listen_first_note_review/bebop_language_note_review.md`
- strict-listen top4 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_strict_listen/listen_first_by_progression/`
- strict-listen top4 note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_strict_listen_note_review/bebop_language_note_review.md`
- strict-listen top4 walking-context 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_walking_context/listen_first_by_progression/`
- strict-listen top4 walking-context note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_walking_context_note_review/bebop_language_note_review.md`
- strict-listen top4 bar-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_bar_repair_probe/listen_first_by_progression/`
- strict-listen top4 bar-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_bar_repair_note_review/bebop_language_note_review.md`
- altered-color balanced 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/listen_first_by_progression/`
- altered-color balanced 전체 WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/audio_with_context/`
- altered-color balanced package report: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/bebop_language_package.md`
- data-contour focused sweep 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/best_listen_first_by_progression/`
- data-contour focused sweep report: `outputs/stage_b_midi_to_solo_bebop_language_package/parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/bebop_language_parameter_sweep.md`
- previous bar-similarity 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v14_bar_similarity_rank/listen_first_by_progression/`
- final review package: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.md`
- review input template: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_review_input_template.json`
- MIDI 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/midi/`
- WAV 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/audio/`
- final review doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- listening input guard doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- MVP handoff freeze doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- listening review pending doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`
- completion audit doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`
- final status sync doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_STATUS_SYNC_2026-06-11.md`
- listening review input wait doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_INPUT_WAIT_2026-06-11.md`

## 재현 명령

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top4_bar_repair_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json' \
  --selected_count 4 \
  --max_per_case 1 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.40625 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json' \
  --selected_count 16 \
  --max_per_case 4 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.46875 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --listen_first_mode consonance
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py \
  --run_id manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance \
  --variants_per_progression 240 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 4010000 \
  --non_chord_probabilities 0.22,0.24,0.26,0.28 \
  --target_chord_tone_ratios 0.78,0.80 \
  --target_offbeat_non_chord_ratios 0.36,0.38,0.40 \
  --max_configs 16
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py \
  --run_id manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution \
  --variants_per_progression 160 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 1010000 \
  --non_chord_probabilities 0.28,0.32,0.34 \
  --target_chord_tone_ratios 0.76,0.78 \
  --target_offbeat_non_chord_ratios 0.38,0.40,0.44 \
  --max_configs 18
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_package.py \
  --run_id manual_2026_06_13_bebop_language_v22_altered_color_balanced \
  --variants_per_progression 1000 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 910000 \
  --non_chord_probability 0.28 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38
```

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

```bash
.venv/bin/python scripts/audit_music_transformer_solo_yield_residual_aware_completion.py \
  --run_id issue_1400_residual_aware_completion_audit \
  --pending_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_review_pending/issue_1398_residual_aware_listening_review_pending/residual_aware_listening_review_pending.json \
  --handoff_freeze_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_mvp_handoff_freeze/issue_1396_residual_aware_mvp_handoff_freeze/residual_aware_mvp_handoff_freeze.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_final_status_sync \
  --require_technical_complete \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/sync_music_transformer_solo_yield_residual_aware_final_status.py \
  --run_id issue_1402_residual_aware_final_status_sync \
  --completion_audit outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_completion_audit/issue_1400_residual_aware_completion_audit/residual_aware_completion_audit.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_listening_review_input_wait \
  --require_final_status_synced \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_input_wait.py \
  --run_id issue_1404_residual_aware_listening_review_input_wait \
  --final_status_sync outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_status_sync/issue_1402_residual_aware_final_status_sync/residual_aware_final_status_sync.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_user_listening_review_fill \
  --require_wait_recorded \
  --require_no_quality_claim
```

## 아닌 것

- 완성형 jazz solo quality claim 아님
- 특정 pianist style adaptation 완료 아님
- production-ready improviser 아님
- human listening preference 검증 완료 아님
- raw audio generation 프로젝트 아님
- bebop language package는 직접 모델 품질 claim이 아니라 청감 수율 개선용 chord-guided comparison package
- parameter sweep는 생성 후보의 음악적 완성도 판정이 아니라 offbeat 해소율과 미해소 비율 기준의 로컬 비교
- chromatic/enclosure/altered 지표는 비밥 장치 proxy이며 human listening preference 대체 아님
- half-repeat 지표는 단순 패턴 반복 proxy이며 실제 청감 반복감 판정 아님
- bar pitch-class similarity 지표는 마디 간 pitch-class 중복 proxy이며 실제 motif 반복감 판정 아님
- best-of package는 기존 후보 재랭킹/재렌더 산출물이며 새 모델 학습 결과 claim 아님

## 주요 파일

- `scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py`
- `scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_package.py`
- `scripts/build_music_transformer_solo_yield_rhythm_syncopation_balance_repair_package.py`
- `scripts/build_music_transformer_solo_yield_residual_aware_final_review_package.py`
- `scripts/guard_music_transformer_solo_yield_residual_aware_listening_input.py`
- `scripts/freeze_music_transformer_solo_yield_residual_aware_mvp_handoff.py`
- `scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_pending.py`
- `scripts/audit_music_transformer_solo_yield_residual_aware_completion.py`
- `scripts/sync_music_transformer_solo_yield_residual_aware_final_status.py`
- `scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_input_wait.py`
- `scripts/build_music_transformer_solo_yield_objective_quality_rubric_baseline.py`
- `scripts/decide_music_transformer_solo_yield_residual_tension_target.py`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_STATUS_SYNC_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_INPUT_WAIT_2026-06-11.md`
