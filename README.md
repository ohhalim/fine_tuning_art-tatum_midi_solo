# Jazz Piano MIDI Solo Generator MVP

입력 MIDI 또는 chord context를 기준으로 짧은 jazz piano solo-line 후보를 만들고,
MIDI/WAV/review package까지 생성하는 symbolic MIDI generation MVP입니다.

현재 목표는 "완성된 고품질 재즈 연주 모델"이 아니라, 입력 context를 받아
재현 가능한 solo 후보를 생성하고 객관 지표로 후보를 고르는 로컬 파이프라인입니다.

## 연구 개요 — 스타일 파인튜닝 병목 진단 (D0→D4)

작은(13.4M) Music Transformer를 특정 재즈 피아니스트 스타일로 파인튜닝하자 생긴
다양성 붕괴·단선율 문제를, **하나의 실험이 하나의 질문에 답하는 통제 실험 5단계**로
역추적했습니다. 핵심 결론: **품질 저하는 base 모델의 표현력 한계가 아니라**, 아래 네
층에 나뉜 원인이었고 각각 처방을 확정했습니다.

| 실험 | 층 | 물음 | 결론 |
|---|---|---|---|
| [D0](docs/d0_experiment/D0_RESULTS.md) | L2 다양성 | base 크기 탓? | 아니오 — **데이터 레짐** (val 5.06→3.19, 학습데이터 상한의 93% 도달) |
| [D1](docs/d1_experiment/D1_RESULTS.md) | L3 스타일 | 붕괴 막는 FT 방식? | pretrain → **LoRA 적응** (다양성 142, 상한의 98%) |
| [D2](docs/d2_experiment/D2_RESULTS.md) | L1 문법 | 깨진 MIDI 문법? | **문법 마스킹**으로 고아 이벤트 0, 밀도 +40% |
| [D3](docs/d3_experiment/D3_RESULTS.md) | L4 음악성 | 단선율 원인? | **조건 프라이머 텍스처 잠김** (진단) |
| [D4](docs/d4_experiment/D4_RESULTS.md) | L4→L5 | 프라이머로 해소? | 화음 프라이머 1.31→1.78 **강한 성공** (처방) |

전체 서사와 기각된 가설 기록은 **[docs/RESEARCH_SUMMARY.md](docs/RESEARCH_SUMMARY.md)**,
층별 로드맵은 [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md)를 보세요.

## 무엇을 만들었나

- 입력: chord progression 또는 guide MIDI context
- 출력: 2-8마디 piano solo-line MIDI 후보
- 부가 출력: solo WAV, context 포함 WAV, note review, handoff report
- 생성 방식: Music Transformer 계열 symbolic checkpoint + constrained decoding
- 후처리: chord-tone landing, offbeat resolution, large-leap, adjacent-repeat, rhythm articulation repair
- 선별 기준: gate penalty, chord-tone ratio, offbeat non-chord ratio, interval repeat, case balance, phrase motion metrics

## 현재 동작 범위

- MIDI tokenization 및 checkpoint 기반 생성 경로
- chord-aware constrained decoding
- 후보별 MIDI grammar validation
- 후보 repair 및 ranking
- FluidSynth 기반 WAV render
- review package 생성
- listening review 전 quality claim 차단

## 스타일 파인튜닝 표준 레시피 (D0/D1 실험 결론, 2026-07)

스타일 모델을 만들 때는 아래 2단계 커리큘럼을 기본으로 한다:

1. **사전학습**: 전체 재즈 코퍼스(2,777조각)로 모델 전체 학습 → 넓은 재즈 분포 습득
2. **LoRA 적응**: base 동결, LoRA 어댑터만 스타일 서브셋으로 학습
   (`train_qlora.py --checkpoint <사전학습ckpt>` — `--train_full_model` 없이)

근거 (통제 실험, 판정 기준 사전 등록):

| 방식 | 생성 다양성 (pooled 고유 보이싱) | 결과 |
|---|---|---|
| 스타일 데이터만 직접 학습 | 81 | 붕괴 ([D0](docs/d0_experiment/D0_RESULTS.md)) |
| 사전학습 → full FT | 108 | 부분 보존 ([D1](docs/d1_experiment/D1_RESULTS.md) Arm C) |
| 사전학습 → full FT + 증강 ×12 | 80 | 재붕괴 (D1 Arm E) |
| **사전학습 → LoRA 적응** | **142** (사전학습 상한 145의 98%) | **채택** (D1 Arm D) |

붕괴의 원인은 데이터 양이 아니라 사전학습 분포 이탈량이므로, base 동결(LoRA)이
구조적 해법이다. 연주자별 어댑터를 같은 base 위에 스왑하는 확장도 이 구조에서 나온다.

## 대표 결과

최신 accepted frontier:

- package: `manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe`
- selected candidates: `8`
- bars / bpm: `8 / 124`
- solo/context WAV: `8 / 8`
- case balance: `2 / 2 / 2 / 2`
- max gate penalty: `0.0000`
- strong-beat chord-tone ratio: `1.0000`
- offbeat non-chord ratio: `0.3672`
- offbeat resolution ratio: `1.0000`
- unresolved offbeat non-chord ratio: `0.0000`
- step motion ratio: `0.4583`
- chromatic step ratio: `0.2599`
- large leap ratio: `0.0337`
- enclosure proxy ratio: `0.3242`
- interval trigram repeat ratio: `0.0123`
- review ready: `true`
- quality claimed: `false`
- model direct claimed: `false`

대표 후보 case:

| rank | case | chords |
|---:|---|---|
| 1 | dominant_cycle | `Em7,A7,Dmaj7,G7` |
| 2 | major_ii_v_turnaround | `Dm7,G7,Cmaj7,A7` |
| 3 | minor_backdoor | `Cm7,F7,Bbmaj7,Ebmaj7` |
| 4 | rhythm_turnaround | `Bbmaj7,G7,Cm7,F7` |

## 주요 산출물

- latest package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.md`
- solo MIDI: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/midi/`
- solo WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/audio/`
- context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/audio_with_context/`
- note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review/bebop_language_note_review.md`
- review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_case_balanced_motion_interval_guard_review_handoff/bebop_language_review_handoff.md`
- detailed status log: `docs/CURRENT_STATUS_AND_PLAN.md`

## 빠른 실행

기본 검증:

```bash
bash scripts/agent_harness.sh quick
```

MVP demo:

```bash
bash scripts/agent_harness.sh demo
```

최신 top8 package 재생성:

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 2 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --max_gate_penalty 1.0 \
  --max_offbeat_non_chord_ratio 0.390625 \
  --max_unresolved_offbeat_non_chord_ratio 0.10 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_interval_trigram_repeat_ratio 0.0164 \
  --max_bar_pitch_class_jaccard 0.675 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 8 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 12 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 24 \
  --target_min_step_motion_ratio 0.43 \
  --target_min_chromatic_step_ratio 0.24 \
  --target_max_large_leap_ratio 0.045 \
  --max_motion_balance_bar_pitch_class_jaccard 0.675 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

note review 생성:

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

review handoff 생성:

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_case_balanced_motion_interval_guard_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

## 최근 개선 요약

- offbeat non-chord ratio: `0.4063 -> 0.3711 -> 0.3672`
- selected case balance: `3/3/1/1 -> 2/2/2/2`
- step motion ratio: `0.4325 -> 0.4583`
- chromatic step ratio: `0.2321 -> 0.2599`
- large leap ratio: `0.0456 -> 0.0337`
- dominant altered offbeat ratio: `0.1719 -> 0.0703`
- interval trigram repeat guard: strict cap `0.0125` selected `8` 불가, feasible cap `0.0164`
- source expansion result: generated/selected `4800 / 128`, strict interval + resolution + unresolved 조건 동시 만족 `0`

## 현재 한계

- human listening preference 미반영
- 고품질 재즈 연주 완성 claim 없음
- model-direct generation quality claim 없음
- 현재 구조: model-conditioned symbolic generation + constrained decoding + repair/ranking hybrid
- strict interval cap `0.0125`와 offbeat/resolution guard 동시 만족 후보 부족
- 다음 판단 대상: listening review input 또는 interval-repeat target 재설계

## 주요 파일

- `scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py`
- `scripts/agent_harness.sh`
- `inference/app/generator.py`
- `inference/app/metrics.py`
- `inference/app/postprocess.py`
- `docs/CURRENT_STATUS_AND_PLAN.md`
