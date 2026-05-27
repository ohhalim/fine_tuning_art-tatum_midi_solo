# Stage B Data-Derived Timing Phrase Vocabulary Repair

작성일: 2026-05-27

## 목적

Issue #162는 Issue #160 proxy review에서 확인한 timing stiffness와 phrase vocabulary blocker를 다시 generation rule 쪽에서 좁게 본 작업이다.

목표:

- penalty-only pitch-cell repair를 더 누적하지 않는다.
- motif extraction의 `top_full_templates`에서 phrase-like timing template을 우선 사용한다.
- register-safe bounds, max interval, final guide/chord landing, objective-clean output은 유지한다.

이 결과는 symbolic MIDI generation probe이며, broad training이나 Brad style adaptation 성공 근거가 아니다.

## 구현

변경 대상:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`

추가한 guard:

- `data_derived_timing_template_rows(...)`
  - `top_full_templates`에서 position span, duration span, IOI variation이 있는 phrase-like timing rows를 먼저 고른다.
  - local phrase slot에 들어가기 어려운 long-span template과 long-sustain template은 제외한다.
  - phrase-like full template이 너무 적으면 기존 rhythm template을 fallback으로 붙인다.

적용 방식:

- `data_motif_rhythm_phrase_variation`의 rhythm row selection만 data-derived full-template 우선순위로 바꿨다.
- position/duration shaping은 기존 review-safe `varied_phrase_positions(...)`, `varied_phrase_duration_tokens(...)`를 유지했다.

제외한 방식:

- full-template position/duration을 그대로 쓰는 pure data-derived timing path도 시험했지만, IOI repetition과 final landing metric이 악화되어 최종 변경에서 제외했다.
- 따라서 이번 변경은 pure timing transplant가 아니라 data-derived row selection + existing review-safe shaping이다.

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

`data_motif_rhythm_phrase_variation` summary:

| metric | Issue #158 | Issue #162 |
|---|---:|---:|
| valid | 3/3 | 3/3 |
| strict | 3/3 | 3/3 |
| final landing resolved | 3/3 | 3/3 |
| max abs interval | 4 | 4 |
| avg syncopated onset ratio | 0.684 | 0.693 |
| avg unique bar-position pattern ratio | 0.958 | 0.958 |
| avg duration diversity ratio | 0.079 | 0.073 |
| avg most-common duration ratio | 0.384 | 0.396 |
| avg IOI diversity ratio | 0.091 | 0.079 |
| avg most-common IOI ratio | 0.385 | 0.392 |
| avg tension ratio | 0.358 | 0.375 |
| avg root-tone ratio | 0.021 | 0.021 |
| objective MIDI flags | `{}` | `{}` |

Top variation candidates:

| candidate | notes | unique pitches | sync | bar-var | dur-var | ioi-var | ioi-rep | tension | landing | max interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 64 | 18 | 0.672 | 1.000 | 0.078 | 0.079 | 0.365 | 0.391 | guide | 4 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 64 | 16 | 0.703 | 1.000 | 0.062 | 0.079 | 0.444 | 0.391 | guide | 4 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 64 | 19 | 0.703 | 0.875 | 0.078 | 0.079 | 0.365 | 0.344 | guide | 4 |

Objective MIDI note review for variation candidates:

- objective bucket: `clean=3`
- objective flags: `[]` for all variation candidates
- note count: `64`
- unique pitch count: `16-19`
- outside ratio: `0.000-0.031`
- max active notes: `1`

## 해석

Improved:

- strict/objective-clean status is preserved.
- final guide/chord landing is restored to `3/3`.
- syncopated onset ratio improves from `0.684` to `0.693`.
- source tension ratio improves from `0.358` to `0.375`.

Tradeoff:

- duration diversity falls from `0.079` to `0.073`.
- IOI diversity falls from `0.091` to `0.079`.
- most-common IOI ratio rises from `0.385` to `0.392`.

Decision:

- Keep the data-derived timing row selection as a reviewable tradeoff, not as a final musical improvement claim.
- Do not promote candidates without a fresh MIDI-note/context proxy review.
- The next issue should judge whether higher syncopation/tension is worth the small IOI/duration regression.

## Next Recommended Issue

```text
Stage B data-derived timing phrase repaired proxy review
```

The next review should decide whether this tradeoff sounds less mechanical, or whether the generator should move to a stronger phrase-level duration/IOI objective instead.

## 검증

실행한 검증:

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
bash scripts/agent_harness.sh quick
```
