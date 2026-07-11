# Stage B Focused Timing Vocabulary Listening Follow-up Repair

작성일: 2026-05-27

## 목적

Issue #196은 Issue #194 focused listening fill에서 남은 `timing=stiff`, `jazz_vocabulary=thin` 병목을 generation rule 쪽에서 좁게 본 작업이다.

중요한 경계:

- 이 작업은 broad training이 아니다.
- focused fill 후보가 final keep으로 승격된 것이 아니다.
- 목표는 max interval, final landing, objective-clean guardrail을 유지하면서 adjacent pitch repeat와 short-cell replay를 줄이는 것이다.

## 배경

Issue #194 결과:

- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- focused listening decision: `needs_followup`
- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `acceptable`
- landing: `strong`
- jazz vocabulary: `thin`

다음 repair target:

- adjacent pitch repeats
- duplicated 3-note pitch-class cells
- grid-derived timing feel
- low source tension / thin chord color

유지해야 하는 guardrail:

- objective-clean status
- safe register range
- final guide/chord landing
- max interval
- no overlap/polyphony
- no duplicated 4-note/8-note pitch-class chunks where possible

## 변경

- `register_safe_phrase_pitch_classes()`에서 safe alternative가 있을 때 직전 pitch-class를 후보에서 제외했다.
- `bounded_phrase_pitch_for_pitch_classes()`에 repeat fallback 직전의 safe color fallback 후보를 추가했다.
- repeat fallback이 필요할 때 최근에 쓰지 않은 실제 pitch를 우선 선택해 unique pitch vocabulary가 무너지지 않게 했다.
- `data_motif_rhythm_phrase_variation`의 line pitch 선택에서 tension, recovery, next guide-tone pitch-class를 repeat fallback 대체 후보로 넘겼다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
RUN_ID=harness_stage_b_focused_timing_vocab_listening_followup_repair bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

## 결과

`data_motif_rhythm_phrase_variation` summary:

| metric | issue #184 | issue #196 |
|---|---:|---:|
| strict valid | `3/3` | `3/3` |
| final landing resolved | `3/3` | `3/3` |
| max interval | `4` | `4` |
| objective flags | `{}` | `{}` |
| avg syncopated onset ratio | `0.703` | `0.703` |
| avg duration diversity ratio | `0.089` | `0.089` |
| avg most-common IOI ratio | `0.397` | `0.397` |
| avg source tension ratio | `0.323` | `0.307` |
| avg root-tone ratio | `0.031` | `0.036` |

Pitch-cell comparison on repaired variation review MIDI:

| candidate rank | issue #184 adjacent repeats | issue #196 adjacent repeats | issue #184 3-cell repeats | issue #196 3-cell repeats | issue #196 4-cell repeats | issue #196 8-cell repeats |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `2` | `0` | `4` | `6` | `2` | `0` |
| 2 | `4` | `0` | `7` | `0` | `0` | `0` |
| 3 | `2` | `0` | `2` | `5` | `0` | `0` |

Objective MIDI review for repaired variation candidates:

| candidate | unique pitches | repeated pitch ratio | objective tension | outside ratio | flags |
|---|---:|---:|---:|---:|---|
| rank 1 | `20` | `0.000` | `0.406` | `0.000` | `[]` |
| rank 2 | `19` | `0.000` | `0.469` | `0.016` | `[]` |
| rank 3 | `20` | `0.000` | `0.453` | `0.000` | `[]` |

## 판단

Issue #196은 adjacent pitch repeat를 세 후보 모두에서 제거했고 objective-clean guardrail을 유지했다.

가장 좋은 신호:

- rank 2는 adjacent repeats `4 -> 0`, 3-cell repeats `7 -> 0`, 4-cell repeats `3 -> 0`으로 개선됐다.
- objective flags는 여전히 없다.
- max interval과 final landing guardrail이 유지됐다.

남은 tradeoff:

- avg source tension ratio가 `0.323 -> 0.307`로 낮아졌다.
- rank 1/3의 duplicated 3-note pitch-class cells는 늘었다.
- 따라서 이 repair는 final keep이 아니라 fresh proxy review 대상이다.

## 다음 작업

`Stage B focused timing vocabulary listening follow-up repaired proxy review`

목표:

- Issue #196 repaired 후보를 MIDI-note/context 기준으로 다시 판단한다.
- rank 2의 repeat/cell improvement가 proxy keep으로 이어지는지 확인한다.
- source tension 하락과 rank 1/3 short-cell tradeoff를 분리한다.
- proxy keep이 없으면 chord-color/timing repair 축을 다시 좁힌다.
