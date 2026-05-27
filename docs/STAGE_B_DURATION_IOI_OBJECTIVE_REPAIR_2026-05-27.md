# Stage B Duration/IOI Objective Repair

Issue #168은 Issue #164 이후 남은 phrase-level duration/IOI 병목을 좁게 본 작업이다.

## 배경

Issue #164 proxy review 결과, `data_motif_rhythm_phrase_variation` 후보는 objective-clean 상태를 유지했지만 proxy `keep` 후보는 만들지 못했다.

직전 수치:

- avg syncopated onset ratio: `0.693`
- avg duration diversity ratio: `0.073`
- avg IOI diversity ratio: `0.079`
- avg most-common IOI ratio: `0.392`
- avg tension ratio: `0.375`
- final landing resolved: `3/3`
- objective flags: `{}`

판단:

- row selection만 더 바꾸지 않는다.
- duration/IOI objective를 직접 개선한다.
- objective-clean guardrail과 final landing guardrail은 유지한다.

## 변경

- `data_motif_rhythm_phrase_variation`의 8-note/bar 경로에 phrase-level duration/IOI bar-position plan을 추가했다.
- 8-bar phrase 안에서 `1-7` step IOI가 나오도록 bar-level position pattern을 분산했다.
- review candidate sort key에서 IOI diversity와 most-common IOI를 duration diversity보다 먼저 보도록 바꿨다.
- 기존 register-safe final cadence, max interval, objective-clean guardrail은 그대로 유지했다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
RUN_ID=harness_stage_b_duration_ioi_objective_repair bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

## 결과

`data_motif_rhythm_phrase_variation` summary:

| metric | previous | issue #168 |
|---|---:|---:|
| strict valid | `3/3` | `3/3` |
| final landing resolved | `3/3` | `3/3` |
| max interval | `4` | `4` |
| objective flags | `{}` | `{}` |
| avg syncopated onset ratio | `0.693` | `0.682` |
| avg duration diversity ratio | `0.073` | `0.078` |
| avg IOI diversity ratio | `0.079` | `0.111` |
| avg most-common IOI ratio | `0.392` | `0.481` |
| avg tension ratio | `0.375` | `0.375` |

Objective MIDI review:

- candidate count: `6`
- objective bucket: `clean=6`
- objective flags: `{}`
- objective reviewable: `6`
- duplicate note sequences: `0`

## 판단

Issue #168은 duration/IOI diversity objective를 개선했다.

다만 most-common IOI ratio가 악화됐기 때문에, 이 변경을 musical keep으로 승격하지 않는다. 다음 단계는 repaired 후보를 MIDI-note/context 기준으로 proxy review해서, 늘어난 IOI 종류가 실제로 덜 mechanical하게 들리는지 확인하는 것이다.

## 다음 작업

`Stage B duration IOI repaired proxy review`
