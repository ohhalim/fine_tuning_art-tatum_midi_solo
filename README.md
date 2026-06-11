# Jazz Piano MIDI Solo Generator MVP

MIDI 데이터를 token sequence로 변환하고, Music Transformer 계열 symbolic model을 학습해 짧은 재즈 솔로 후보 MIDI를 생성하는 로컬 MVP.

현재 목표는 완성형 재즈 연주 모델이 아니다. 목표는 **학습된 checkpoint를 사용해 2-4마디 솔로 후보 MIDI/WAV를 만들고, 재즈 솔로 후보로 남길 수 있는 결과의 수율을 높이는 것**이다.

## 현재 완료 범위

- Music Transformer 계열 checkpoint 기반 constrained generation 경로 동작 확인
- 2마디 후보 생성: strict `24 / 24`, grammar-valid `24 / 24`
- 4마디 확장 후보 생성: strict `20 / 24`, grammar-valid `24 / 24`
- 4마디 dead-air repair 이후: strict `22 / 24`, grammar-valid `24 / 24`
- sampling repeatability audit: grammar `12 / 12`, strict `8 / 12`, failing case `rhythm_turnaround`
- final status audit: technical evidence ready `true`
- 음악적 품질 claim: `false`
- 사람 기준 청취 선호 입력: `false`

## 무엇을 만들었나

- Stage B tokenized MIDI dataset 기반 Music Transformer 계열 모델 학습
- 학습 checkpoint 저장
- chord context 기반 2-4마디 솔로 후보 생성
- model logits 기반 token sampling
- MIDI 문법 붕괴를 줄이기 위한 constrained decoding
- 후보별 objective metric 계산
- 상위 MIDI 후보 export
- WAV 렌더링

## 왜 constrained decoding을 썼나

raw model generation은 note grammar가 자주 깨졌다.

- note가 거의 없는 MIDI
- note_on/note_off 구조 붕괴
- 반복 또는 빈 구간 증가
- MIDI review gate 통과 실패

그래서 모델을 버린 것이 아니라, **모델 checkpoint의 logits로 token을 고르되 MIDI 문법과 chord/rhythm 범위를 제한**했다. 이 방식은 규칙만으로 MIDI를 만든 것이 아니라, 학습된 모델의 확률분포를 사용한 constrained generation이다.

## 현재 확인 결과

사용 checkpoint:

- `outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/training_smoke/controlled_2048_512_maxseq160/checkpoints/checkpoint_epoch1.pt`

학습 조건:

- train records: `2048`
- val records: `512`
- train loss: `3.8923`
- validation loss: `3.0396`
- model: `n_layers=1`, `d_model=64`, `max_sequence=160`

생성 조건:

- chords: `Cm7, Fm7, Bb7, Ebmaj7`
- bars: `2`
- bpm: `124`
- candidates: `20`
- decoding: `constrained`
- pitch: chord-aware `approach_tensions`
- rhythm: `swing_motif`

수율:

- grammar-valid candidates: `20 / 20`
- MIDI review valid candidates: `18 / 20`
- strict valid candidates: `18 / 20`
- selected MIDI candidates: `5`
- rendered WAV files: `5`

반복 스윕:

- chord progression cases: `4`
- total candidates: `24`
- grammar-valid candidates: `24 / 24`
- strict valid candidates: `18 / 24`
- rendered WAV files: `8`
- lowest case: `Dm7, G7, Cmaj7, A7` strict `2 / 6`
- lowest case review: invalid `4 / 4` dead-air threshold miss, next target `duration_fill_or_overlap_aftercare`
- dead-air repair: `fill_n10` 기준 strict `2 / 6 -> 6 / 6`, dead-air fail `4 -> 0`
- repaired full sweep: `fill_n10` 기준 strict `24 / 24`, grammar-valid `24 / 24`, rendered WAV `8`
- listening review package: MIDI `8`, WAV `8`, review input template ready
- listening input guard: validated input `false`, preference fill `false`, next `objective_only_next_decision`
- objective-only decision: candidate `8`, selected objective candidates `4`, musical quality claim `false`
- next boundary: `music_transformer_solo_yield_larger_sample_repeatability_sweep`
- larger sample repeatability: total candidates `48`, strict `47 / 48`, grammar-valid `48 / 48`
- larger sample min case strict rate: `0.9167`, rendered WAV files `12`, musical quality claim `false`
- larger sample listening package: MIDI `12`, WAV `12`, review input template ready
- larger sample input guard: validated input `false`, preference fill `false`, pending candidate fields `36`
- next boundary: `music_transformer_solo_yield_objective_only_next_decision`
- larger sample objective-only decision: candidate `12`, selected objective candidates `6`, musical quality claim `false`
- next boundary: `music_transformer_solo_yield_4bar_phrase_expansion_probe`
- 4bar phrase expansion: total candidates `24`, strict `20 / 24`, grammar-valid `24 / 24`
- 4bar min case strict rate: `0.6667`, rendered WAV files `8`, musical quality claim `false`
- 4bar listening package: MIDI `8`, WAV `8`, review input template ready
- 4bar input guard: validated input `false`, preference fill `false`, pending candidate fields `24`
- next boundary: `music_transformer_solo_yield_objective_only_next_decision`
- 4bar objective-only decision: candidate `8`, selected objective candidates `4`, dead-air range `0.6552 - 0.7692`
- next boundary: `music_transformer_solo_yield_4bar_dead_air_repair_sweep`
- 4bar dead-air repair: strict `20 / 24 -> 22 / 24`, case avg dead-air range `0.7171 - 0.7571 -> 0.6340 - 0.6545`
- rejected repair variant: `note_groups_per_bar=10`, `max_sequence=192`, reason checkpoint `model_max_sequence=160`
- 4bar repaired listening package: MIDI `8`, WAV `8`, review input template ready
- 4bar repaired input guard: validated input `false`, preference fill `false`, pending candidate fields `24`
- next boundary: `music_transformer_solo_yield_objective_only_next_decision`
- 4bar repaired objective-only decision: candidate `8`, selected objective candidates `4`, dead-air range `0.5152 - 0.7241`
- next boundary: `music_transformer_solo_yield_final_status_audit`
- final status audit: technical evidence ready `true`, strict `22 / 24`, grammar-valid `24 / 24`, rendered WAV `8`
- final status audit claim boundary: musical quality claim `false`, raw artifact upload required `false`
- next boundary: `music_transformer_solo_yield_readme_final_evidence_refresh`
- 4bar repaired top8 objective failure review: final landing not chord-tone `8 / 8`, package low chord-tone ratio `8 / 8`, MIDI low chord-tone ratio `6 / 8`, dead-air still high `3 / 8`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_sweep`
- chord-tone landing repair: repaired MIDI `8`, changed note `8`, max pitch shift `2`, final landing not chord-tone `8 / 8 -> 0 / 8`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_audio_package`
- chord-tone landing repair audio package: rendered WAV `8`, technical WAV validation `true`, duration range `10.725s - 10.739s`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_listening_package`
- chord-tone landing repair listening package: MIDI `8`, WAV `8`, review input template `true`, validated listening input `false`, preference fill `false`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_listening_input_guard`
- chord-tone landing repair listening input guard: schema matched `true`, pending candidate fields `24`, objective-only next decision required `true`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_objective_only_next_decision`
- chord-tone landing repair objective-only decision: final landing residual `0 / 8`, weak direction-change `4 / 8`, MIDI low chord-tone ratio `3 / 8`, dead-air aftercare `0 / 8`
- next boundary: `music_transformer_solo_yield_phrase_direction_repair_sweep`
- phrase direction repair sweep: repaired MIDI `8`, weak direction-change `4 / 8 -> 0 / 8`, changed note `5`, chord-tone ratio decrease `0`, final landing residual `0`
- next boundary: `music_transformer_solo_yield_phrase_direction_repair_audio_package`
- phrase direction repair audio package: rendered WAV `8`, technical WAV validation `true`, duration range `10.725s - 10.739s`
- next boundary: `music_transformer_solo_yield_phrase_direction_repair_listening_package`
- phrase direction repair listening package: MIDI `8`, WAV `8`, review input template `true`, validated listening input `false`, preference fill `false`
- next boundary: `music_transformer_solo_yield_phrase_direction_repair_listening_input_guard`
- phrase direction repair listening input guard: schema matched `true`, pending candidate fields `24`, objective-only next decision required `true`
- next boundary: `music_transformer_solo_yield_phrase_direction_repair_objective_only_next_decision`
- phrase direction repair objective-only decision: weak direction residual `0 / 8`, MIDI low chord-tone ratio `3 / 8`, low note count `2 / 8`, wide interval review `2 / 8`
- next boundary: `music_transformer_solo_yield_chord_role_balance_repair_sweep`
- chord role balance repair sweep: low chord-role `3 / 8 -> 0 / 8`, changed note `3`, max pitch shift `2`, weak direction residual `0 / 8`, final landing residual `0 / 8`, wide interval review `2 / 8 -> 1 / 8`
- next boundary: `music_transformer_solo_yield_chord_role_balance_repair_audio_package`
- chord role balance repair audio package: rendered WAV `8`, technical WAV validation `true`, duration range `10.725s - 10.739s`
- next boundary: `music_transformer_solo_yield_chord_role_balance_repair_listening_package`
- chord role balance repair listening package: MIDI `8`, WAV `8`, review input template `true`, validated listening input `false`, preference fill `false`
- next boundary: `music_transformer_solo_yield_chord_role_balance_repair_listening_input_guard`
- chord role balance repair listening input guard: schema matched `true`, pending candidate fields `24`, objective-only next decision required `true`
- next boundary: `music_transformer_solo_yield_chord_role_balance_repair_objective_only_next_decision`
- chord role balance repair objective-only decision: MIDI low chord-tone ratio `0 / 8`, low note count `2 / 8`, wide interval review `1 / 8`, weak direction residual `0 / 8`, final landing residual `0 / 8`
- next boundary: `music_transformer_solo_yield_density_aftercare_sweep`
- density aftercare sweep: low note count `2 / 8 -> 0 / 8`, inserted note `3`, chord-tone ratio decrease `0`, weak direction residual `0 / 8`, final landing residual `0 / 8`, wide interval review `1 / 8 -> 1 / 8`
- next boundary: `music_transformer_solo_yield_density_aftercare_audio_package`
- density aftercare audio package: rendered WAV `8`, technical WAV validation `true`, duration range `10.725s - 10.739s`
- next boundary: `music_transformer_solo_yield_density_aftercare_listening_package`
- density aftercare listening package: MIDI `8`, WAV `8`, review input template `true`, validated listening input `false`, preference fill `false`
- next boundary: `music_transformer_solo_yield_density_aftercare_listening_input_guard`
- density aftercare listening input guard: schema matched `true`, pending candidate fields `24`, objective-only next decision required `true`
- next boundary: `music_transformer_solo_yield_density_aftercare_objective_only_next_decision`
- density aftercare objective-only decision: MIDI low chord-tone ratio `0 / 8`, low note count `0 / 8`, wide interval review `1 / 8`, dead-air aftercare `0 / 8`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_sweep`
- interval contour aftercare sweep: wide interval review `1 / 8 -> 0 / 8`, adjusted note `1`, max pitch shift `1`, guard regression `0`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_audio_package`
- interval contour aftercare audio package: rendered WAV `8`, technical WAV validation `true`, duration range `10.725s - 10.739s`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_listening_package`
- interval contour aftercare listening package: MIDI `8`, WAV `8`, review input template `true`, validated listening input `false`, preference fill `false`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_listening_input_guard`
- interval contour aftercare listening input guard: schema matched `true`, pending candidate fields `24`, objective-only next decision required `true`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_objective_only_next_decision`
- interval contour aftercare objective-only decision: MIDI low chord-tone ratio `0 / 8`, low note count `0 / 8`, wide interval review `0 / 8`, dead-air aftercare `0 / 8`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_listening_review`
- interval contour final review handoff: MIDI `8`, WAV `8`, objective residual labels `0`, preference fill `false`
- next boundary: `music_transformer_solo_yield_interval_contour_aftercare_listening_review`
- interval contour handoff reproducibility audit: missing MIDI/WAV `0 / 0`, checksum mismatch `0 / 0`, reproducible handoff `true`
- next boundary: `music_transformer_solo_yield_sampling_repeatability_audit`
- sampling repeatability audit: grammar `12 / 12`, strict `8 / 12`, min case strict rate `0.3333`
- failing case: `rhythm_turnaround`, strict `1 / 3`
- next boundary: `music_transformer_solo_yield_failure_case_review`

## 결과 파일

최신 리뷰 패키지:

- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_final_handoff/issue_1314_interval_contour_final_handoff/interval_contour_final_review_handoff.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_final_handoff/issue_1314_interval_contour_final_handoff/interval_contour_final_review_handoff.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_handoff_audit/issue_1316_interval_contour_handoff_audit/interval_contour_handoff_reproducibility_audit.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_handoff_audit/issue_1316_interval_contour_handoff_audit/interval_contour_handoff_reproducibility_audit.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/solo_yield_sweep_report.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/solo_yield_sweep_report.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_listening_review/issue_1308_interval_contour_listening_package/listening_review_package.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_listening_review/issue_1308_interval_contour_listening_package/listening_review_package.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_listening_review/issue_1308_interval_contour_listening_package/listening_review_input_template.json`

MIDI 후보:

- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_listening_review/issue_1308_interval_contour_listening_package/midi/`

WAV 후보:

- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_listening_review/issue_1308_interval_contour_listening_package/audio/`

Report:

- `outputs/music_transformer_finetune_mvp/solo_yield_final_status_audit/issue_1256_final_status_audit/final_status_audit.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_final_status_audit/issue_1256_final_status_audit/final_status_audit_summary.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_objective_next_decision/issue_1254_4bar_repaired_objective_next_decision/objective_next_decision.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_aftercare_objective_next/issue_1312_interval_contour_objective_next/interval_contour_aftercare_objective_next_decision.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_final_handoff/issue_1314_interval_contour_final_handoff/interval_contour_final_review_handoff.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_interval_contour_handoff_audit/issue_1316_interval_contour_handoff_audit/interval_contour_handoff_reproducibility_audit.md`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/solo_yield_sweep_report.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_PROGRESSION_YIELD_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_YIELD_FAILURE_CASE_REVIEW_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DEAD_AIR_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_REPAIRED_PROGRESSION_RETRY_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_REPAIRED_CANDIDATE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_REPEATABILITY_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_PHRASE_EXPANSION_PROBE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_DEAD_AIR_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_REPAIRED_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_REPAIRED_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_4BAR_REPAIRED_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_FINAL_STATUS_AUDIT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_README_FINAL_EVIDENCE_REFRESH_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_FINAL_HANDOFF_SUMMARY_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_REPAIRED_TOP8_OBJECTIVE_FAILURE_REVIEW_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_TONE_LANDING_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_TONE_LANDING_REPAIR_AUDIO_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_TONE_LANDING_REPAIR_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_TONE_LANDING_REPAIR_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_TONE_LANDING_REPAIR_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_PHRASE_DIRECTION_REPAIR_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_ROLE_BALANCE_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_ROLE_BALANCE_REPAIR_AUDIO_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_ROLE_BALANCE_REPAIR_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_ROLE_BALANCE_REPAIR_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_ROLE_BALANCE_REPAIR_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DENSITY_AFTERCARE_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DENSITY_AFTERCARE_AUDIO_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DENSITY_AFTERCARE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DENSITY_AFTERCARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DENSITY_AFTERCARE_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_AFTERCARE_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_AFTERCARE_AUDIO_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_AFTERCARE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_AFTERCARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_AFTERCARE_OBJECTIVE_NEXT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_FINAL_REVIEW_HANDOFF_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_INTERVAL_CONTOUR_HANDOFF_REPRODUCIBILITY_AUDIT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_SAMPLING_REPEATABILITY_AUDIT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_PHRASE_DIRECTION_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_PHRASE_DIRECTION_REPAIR_AUDIO_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_PHRASE_DIRECTION_REPAIR_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_PHRASE_DIRECTION_REPAIR_LISTENING_INPUT_GUARD_2026-06-11.md`

## 실행 방법

20개 후보 생성:

```bash
.venv/bin/python scripts/run_stage_b_generation_probe.py \
  --output_root outputs/music_transformer_finetune_mvp/stage_b_solo_yield_probe \
  --run_id constrained_chord_swing_20 \
  --checkpoint_dir outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/training_smoke/controlled_2048_512_maxseq160/checkpoints \
  --skip_prepare \
  --skip_train \
  --generation_mode constrained \
  --num_samples 20 \
  --seed 700 \
  --bpm 124 \
  --bars 2 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --density medium \
  --temperature 0.85 \
  --top_k 8 \
  --max_sequence 160 \
  --constrained_note_groups_per_bar 8 \
  --coverage_aware_positions \
  --coverage_position_window 1 \
  --chord_aware_pitches \
  --chord_pitch_mode approach_tensions \
  --chord_pitch_repeat_window 2 \
  --constrained_pitch_min 55 \
  --constrained_pitch_max 84 \
  --constrained_max_adjacent_interval 7 \
  --jazz_rhythm_positions \
  --jazz_duration_tokens \
  --jazz_rhythm_profile swing_motif \
  --cap_duration_to_next_position \
  --avoid_reused_positions \
  --postprocess_overlap \
  --max_simultaneous_notes 1
```

상위 후보 패키징:

```bash
.venv/bin/python scripts/build_music_transformer_solo_yield_package.py \
  --probe_report outputs/music_transformer_finetune_mvp/stage_b_solo_yield_probe/constrained_chord_swing_20/report.json \
  --output_root outputs/music_transformer_finetune_mvp/solo_yield_mvp \
  --run_id constrained_chord_swing_top5 \
  --top_n 5
```

## 현재 한계

- 사람 기준으로 좋은 재즈 솔로라고 검증된 상태 아님
- Coltrane, Art Tatum, Oscar Peterson 수준의 긴 솔로 생성 아님
- Brad style adaptation 완료 아님
- 입력 MIDI 전체를 이해해 자유롭게 솔로를 만드는 단계 아님
- 현재는 chord context와 constrained decoding을 사용한 2-4마디 후보 생성 MVP

## 다음 작업

- phrase direction repair objective-only next decision
- pending input 기준 다음 repair target 분리
- WAV/MIDI 청취 리뷰
- 청취 결과 기준 keep/reject 후보 기록
- 음악적 품질 claim 여부는 청취 리뷰 이후 재판단
- 다음 품질 개선 후보: phrase tension/release, chord-role balance, outside-soloing pitch-role
