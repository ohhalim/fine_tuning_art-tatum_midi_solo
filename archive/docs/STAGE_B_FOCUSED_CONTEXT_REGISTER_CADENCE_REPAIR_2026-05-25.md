# Stage B Focused Context Register-Cadence Repair

작성일: 2026-05-25

## 목적

Issue #142는 Issue #140 focused context decision에서 확인한 register/cadence blocker를 generation rule 쪽에서 좁게 고친 작업이다.

Blocker:

- 이전 top proxy seed는 bar 4 부근에서 `C6`까지 올라간 뒤 final bar에서 `G3`까지 내려왔다.
- context MIDI 안에서는 final two bars가 bass/root guide register에 가까워져 solo-line cadence보다 low-register exercise처럼 읽혔다.

목표:

- 기존 rhythm/position guardrail은 유지한다.
- objective-clean 상태, duplicate-free 상태, max interval guardrail을 유지한다.
- final cadence를 right-hand solo register 안에 남긴다.
- broad training이나 style adaptation claim으로 확장하지 않는다.

## 구현

변경 파일:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`

추가한 helper:

- `focused_context_register_bounds()`

적용:

- `data_motif_rhythm_phrase_variation`의 기본 pitch range를 `55-79`로 좁혔다.
- 마지막 3 bars는 `58-77` 안으로 제한한다.
- 마지막 2 bars는 `60-76` 안으로 제한한다.
- pending recovery, final landing, phrase target shaping, bounded pitch selection 모두 같은 bar-local range를 사용한다.

이 방식은 pitch를 나중에 강제로 transpose하지 않고, 기존 max interval `4` 선택 과정 안에서 register arc를 제한한다.

## Result

실행:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Summary:

| field | result |
|---|---:|
| variation valid samples | `3/3` |
| variation strict samples | `3/3` |
| final landing resolved | `3/3` |
| max interval | `4` |
| duplicate note sequences | `0` |
| objective MIDI flag counts | `{}` |

Variation aggregate:

| metric | result |
|---|---:|
| avg syncopated onset ratio | `0.684` |
| avg unique bar-position pattern ratio | `0.958` |
| avg duration diversity ratio | `0.079` |
| avg most-common duration ratio | `0.384` |
| avg IOI diversity ratio | `0.091` |
| avg most-common IOI ratio | `0.385` |
| avg tension ratio | `0.395` |
| avg root tone ratio | `0.011` |

Top candidate after repair:

| field | value |
|---|---|
| candidate | `data_motif_rhythm_phrase_variation_rank_1_sample_3` |
| note count | `63` |
| unique pitch count | `18` |
| objective pitch range | `61-79` |
| objective flags | `[]` |
| final landing | `G4` |
| final landing role | `guide` |
| final bar notes | `F4, G4, A#4, A4, F4, D4, F#4, G4` |

Tradeoff:

- unique pitch count fell from the previous top candidate's `28` to `18`.
- this is acceptable for this issue because the target was register/cadence repair, but the next review must check whether the narrower register now sounds too boxed-in.

## Decision

Issue #142 conclusion:

- Keep the register/cadence repair.
- The previous C6-to-G3 focused context blocker is repaired for the top candidate.
- The repair preserves objective-clean status and duplicate-free status.
- This is not yet a musical pass; it needs a new focused proxy review.

Recommended next issue:

```text
Stage B register-cadence repaired focused proxy review
```

Target:

- rebuild focused proxy review evidence from the repaired `harness_stage_b_rhythm_phrase_variation` outputs
- compare the repaired top candidate against Issue #140 blocker notes
- decide whether the narrower register range creates a new boxed-in phrase problem

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py tests/test_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
bash scripts/agent_harness.sh quick
```

Quick harness result:

- unit tests: `236` passed
- compile checks: passed
- diff whitespace check: passed
