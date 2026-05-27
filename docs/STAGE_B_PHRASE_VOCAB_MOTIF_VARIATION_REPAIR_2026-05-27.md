# Stage B Phrase Vocabulary/Motif Variation Repair

Issue #172는 Issue #170 proxy review에서 남은 small-cell mechanical contour와 motif variation 병목을 좁게 본 작업이다.

## 배경

Issue #170은 Issue #168 duration/IOI objective repair 후보를 MIDI-note/context 기준으로 다시 봤지만 proxy `keep` 후보를 만들지 못했다.

직전 판단:

- objective bucket: `clean=6`
- objective flags: `{}`
- decisions: `keep=0`, `needs_followup=4`, `reject=2`
- strongest follow-up: `improve_phrase_vocabulary`
- secondary follow-ups: `fix_timing_grid`, `increase_motif_variation`

Issue #168의 generation summary:

- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- avg duration diversity ratio: `0.078`
- avg IOI diversity ratio: `0.111`
- avg most-common IOI ratio: `0.481`
- avg tension ratio: `0.375`

판단:

- objective-clean guardrail은 유지한다.
- most-common IOI 집중과 최근 pitch 재사용을 같이 줄인다.
- max interval, register-safe final landing, solo-line constraints는 열지 않는다.

## 변경

- `phrase_level_duration_ioi_bar_positions()`의 8-note/bar position pattern을 다시 균형화했다.
- 8-bar phrase에서 most-common IOI가 특정 1-step에 몰리지 않도록 bar pattern 조합을 조정했다.
- `bounded_phrase_pitch_for_pitch_classes()`에 최근 exact pitch와 pitch-class 재사용 penalty를 추가했다.
- normal phrase pitch 선택에서 3/4-step motif-sized interval을 선호하되, 최근 pitch 재사용 회피를 interval 선호보다 먼저 보도록 정렬했다.
- 관련 단위 테스트에 balanced IOI 분포와 preferred interval 선택 검증을 추가했다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
RUN_ID=harness_stage_b_phrase_vocab_motif_variation_repair bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

## 결과

`data_motif_rhythm_phrase_variation` summary:

| metric | issue #168 | issue #172 |
|---|---:|---:|
| strict valid | `3/3` | `3/3` |
| final landing resolved | `3/3` | `3/3` |
| max interval | `4` | `4` |
| objective flags | `{}` | `{}` |
| avg syncopated onset ratio | `0.682` | `0.703` |
| avg duration diversity ratio | `0.078` | `0.089` |
| avg most-common duration ratio | `0.479` | `0.406` |
| avg IOI diversity ratio | `0.111` | `0.095` |
| avg most-common IOI ratio | `0.481` | `0.397` |
| avg tension ratio | `0.375` | `0.318` |
| avg root-tone ratio | `0.016` | `0.047` |

Per-sample repaired variation:

| sample | unique pitch count | final landing | duration diversity | most-common IOI |
|---|---:|---|---:|---:|
| 1 | `19` | `guide` | `0.094` | `0.397` |
| 2 | `18` | `guide` | `0.094` | `0.397` |
| 3 | `20` | `chord_tone` | `0.078` | `0.397` |

Objective MIDI review:

- candidate count: `6`
- objective bucket: `clean=6`
- objective flags: `{}`
- objective reviewable: `6`
- repaired variation unique pitch count: `18-20`
- repaired variation objective tension ratio: `0.438-0.500`

## 판단

Issue #172는 Issue #168의 가장 나쁜 tradeoff였던 most-common IOI 집중을 줄였다.

대신 avg IOI diversity ratio는 `0.111 -> 0.095`로 낮아졌고 source tension ratio도 `0.375 -> 0.318`로 내려갔다. 따라서 이 결과를 final musical keep으로 주장하지 않는다.

현재 해석:

- objective-clean guardrail은 유지됐다.
- duration/IOI 반복 집중은 줄었다.
- pitch vocabulary 폭은 repaired variation 기준 `18-20`개로 회복됐다.
- 그러나 실제 phrase quality는 fresh proxy review로 다시 판단해야 한다.

## 다음 작업

`Stage B phrase vocabulary motif variation repaired proxy review`
