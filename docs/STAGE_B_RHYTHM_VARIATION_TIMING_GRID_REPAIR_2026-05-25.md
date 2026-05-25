# Stage B Rhythm Variation Timing-Grid Repetition Repair

작성일: 2026-05-25

## 목적

Issue #126은 Issue #124 sample-diverse MIDI-note proxy review에서 남은 timing stiffness 문제를 좁혀서 고친다.

Issue #124 결과:

- `keep`: `0`
- `needs_followup`: `6`
- `too_stiff`: `6`
- aggregate:
  - `improve_phrase_vocabulary`: `14`
  - `fix_timing_grid`: `12`
  - `increase_motif_variation`: `6`

이번 목표:

- `data_motif_rhythm_phrase_variation`의 deterministic IOI/rest-template repetition을 줄인다.
- duplicate note sequence count는 `0`으로 유지한다.
- final guide landing, max interval bound, objective-clean MIDI review를 유지한다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `varied_phrase_positions()` 추가
  - variation mode에서 motif 내부 position을 seed/bar/motif별 anti-repetition pattern으로 재배치
  - slot split pattern을 좁은 `5-6` step cell 대신 `7-9` step cell 중심으로 제한
  - variation mode interval bound를 `4`로 조정
  - non-final bar landing은 strict guide cadence보다 bounded phrase pitch를 우선해 review MIDI의 unresolved large-leap risk를 줄임
  - `bounded_phrase_pitch_for_pitch_classes()`와 `cadence_landing_pitch()`에 `allow_wider_fallback` 옵션 추가
- `tests/test_stage_b_data_motif_generation_compare.py`
  - `varied_phrase_positions()`가 adjacent grid cluster를 만들지 않는지 검증
  - rhythm variation candidate의 max interval guard를 `4`로 강화
  - seed별 independent sequence test 유지

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

출력:

- compare report:
  - `outputs/stage_b_data_motif_compare/harness_stage_b_rhythm_phrase_variation/data_motif_compare_report.md`
- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## 결과

Compare summary:

| mode | samples | strict | landing | max interval | sync | bar-var | dur-var | ioi-var | ioi-rep | objective flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `data_motif_contour_landing_repair` | 3 | 3 | 3/3 | 7 | 0.625 | 1.000 | 0.062 | 0.079 | 0.429 | `{}` |
| `data_motif_rhythm_phrase_variation` | 3 | 3 | 3/3 | 4 | 0.695 | 0.583 | 0.084 | 0.070 | 0.412 | `{}` |

Review export duplicate summary:

| metric | value |
|---|---:|
| candidate count | 6 |
| unique note sequences | 6 |
| duplicate note sequences | 0 |

Variation candidates:

| candidate | notes | pitches | sync | bar-var | dur-var | ioi-var | ioi-rep | landing | max interval |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 59 | 25 | 0.683 | 0.750 | 0.095 | 0.065 | 0.484 | guide | 4 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 59 | 24 | 0.762 | 0.500 | 0.079 | 0.081 | 0.371 | guide | 4 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 63 | 25 | 0.641 | 0.500 | 0.078 | 0.063 | 0.381 | guide | 4 |

Objective MIDI review for variation candidates:

- objective flags: `{}`
- large leap ratio: `0.000`
- unresolved large leap ratio: `0.000`
- repeated pitch interval ratio: `0.000`
- pitch ranges:
  - rank 1: `48-81`
  - rank 2: `48-82`
  - rank 3: `50-82`

## 해석

Improved:

- average most-common IOI ratio:
  - before sample-diverse review: `0.497`
  - after timing repair: `0.412`
- max interval:
  - before timing repair: up to `6`
  - after timing repair: `4`
- objective unresolved large-leap risk in variation review MIDI:
  - before timing repair: present in some ranked candidates
  - after timing repair: `0.000`
- duplicate note sequence count remains `0`

Tradeoff:

- IOI diversity fell from `0.108` to `0.070`.
- bar-position variation fell from `1.000` to `0.583`.
- duration diversity fell from `0.106` to `0.084`.

This means the repair reduces dominant IOI repetition and contour risk, but it may have made the position vocabulary more conservative. It is not a musical success claim.

## Decision

Issue #126 is a valid timing-grid repetition repair, with tradeoffs.

Recommended next issue:

```text
Stage B timing-grid repaired rhythm MIDI-note proxy review
```

Next review should decide whether the lower IOI repetition is perceptually better, or whether the lower IOI/bar-position diversity makes the phrase more mechanical.

Do not move to broad training, backend/UI, or audio pivot yet.

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py tests/test_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Commit 전 필수 quick harness도 실행한다.
