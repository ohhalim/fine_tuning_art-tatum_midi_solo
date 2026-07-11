# Stage B Focused Timing Vocabulary Follow-up Repair

작성일: 2026-05-27

## 목적

Issue #184는 Issue #182 focused listening fill에서 드러난 `timing=stiff`, `jazz_vocabulary=thin` 병목을 generation rule 쪽에서 좁게 본 작업이다.

중요한 경계:

- 이 작업은 broad training이 아니다.
- focused fill 후보가 final keep으로 승격된 것이 아니다.
- 목표는 objective-clean/register/cadence guardrail을 유지하면서 short pitch-class cell 반복을 줄일 수 있는지 확인하는 것이다.

## 배경

Issue #182 결과:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- focused listening decision: `needs_followup`
- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `acceptable`
- landing: `acceptable`
- jazz vocabulary: `thin`

다음 repair target:

- repeated 3-note/4-note pitch-class cells
- grid-derived timing feel
- thin/mechanical phrase vocabulary

유지해야 하는 guardrail:

- objective-clean status
- safe register range
- final guide/chord landing
- max interval
- no overlap/polyphony
- no off-grid artifacts

## 변경

- `register_safe_phrase_pitch_classes()`에서 최근 2/3개 pitch-class prefix가 과거 3/4-note cell을 재생하는 candidate pitch-class를 safe alternative가 있을 때 제외했다.
- `register_safe_phrase_cell_penalty()`의 repeated cell penalty를 강화했다.
- `bounded_phrase_pitch_for_pitch_classes()`에서 `allow_wider_fallback=False`인 상황에 max interval 후보가 없으면 넓은 leap 대신 repeat fallback을 허용해 max interval guardrail을 지키도록 했다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
RUN_ID=harness_stage_b_focused_timing_vocab_followup_repair bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

## 결과

`data_motif_rhythm_phrase_variation` summary:

| metric | issue #172 | issue #184 |
|---|---:|---:|
| strict valid | `3/3` | `3/3` |
| final landing resolved | `3/3` | `3/3` |
| max interval | `4` | `4` |
| objective flags | `{}` | `{}` |
| avg syncopated onset ratio | `0.703` | `0.703` |
| avg duration diversity ratio | `0.089` | `0.089` |
| avg most-common duration ratio | `0.406` | `0.406` |
| avg IOI diversity ratio | `0.095` | `0.095` |
| avg most-common IOI ratio | `0.397` | `0.397` |
| avg tension ratio | `0.318` | `0.323` |
| avg root-tone ratio | `0.047` | `0.031` |

Objective MIDI review for repaired variation candidates:

| metric | issue #172 | issue #184 |
|---|---:|---:|
| unique pitch count | `18-20` | `19-20` |
| stepwise interval ratio | `0.460-0.492` | `0.460` |
| repeated pitch interval ratio | `0.000-0.016` | `0.032-0.063` |
| objective tension ratio | `0.438-0.500` | `0.438-0.469` |
| objective flags | `{}` | `{}` |

Pitch-class cell comparison on repaired variation review MIDI:

| candidate rank | issue #172 3-cell repeats | issue #184 3-cell repeats | issue #172 4-cell repeats | issue #184 4-cell repeats | issue #184 8-cell repeats |
|---:|---:|---:|---:|---:|---:|
| 1 | `5` | `4` | `3` | `1` | `0` |
| 2 | `5` | `7` | `2` | `3` | `0` |
| 3 | `5` | `2` | `0` | `0` | `0` |

## 판단

Issue #184는 objective-clean guardrail을 유지했고, rank 1/3에서는 short pitch-class cell repetition을 줄였다.

그러나 rank 2에서는 repeated short cells와 adjacent pitch repeat이 늘었다. 따라서 이 repair는 final keep으로 승격하지 않는다.

현재 해석:

- max interval, final landing, objective flags는 유지됐다.
- root-tone ratio와 stepwise ratio는 개선됐다.
- repeated pitch interval이 생겼기 때문에 fresh proxy review가 필요하다.
- 다음 판단은 candidate rank 3까지 포함한 repaired set을 MIDI-note/context 기준으로 다시 채우는 것이다.

## 다음 작업

`Stage B focused timing vocabulary repaired proxy review`
