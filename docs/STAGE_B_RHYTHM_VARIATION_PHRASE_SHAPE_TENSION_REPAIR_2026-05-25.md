# Stage B Rhythm Variation Phrase-Shape Tension Repair

작성일: 2026-05-25

## 목적

Issue #134는 Issue #132 proxy review에서 남은 no-keep 병목을 generation rule 쪽에서 다시 좁힌 작업이다.

목표:

- Issue #130 position/IOI guardrail을 유지한다.
- high-register phrase sketch와 safe scalar exercise 문제를 줄인다.
- tension/approach color를 늘린다.
- outside-note, unresolved-leap, duplicate, overlap/polyphony objective flags를 만들지 않는다.

## 변경

`data_motif_rhythm_phrase_variation`에 다음을 추가했다.

- `phrase_shape_target_pitch()`:
  - raw target pitch를 seeded phrase register center 쪽으로 당겨 extreme high-register sketch를 줄인다.
- `phrase_shape_pitch_classes()`:
  - 일부 local phrase role에서 current chord tension pitch class를 우선한다.
  - 일부 phrase role에서 next chord guide tone을 우선한다.
  - 기존 cadence/phrase recovery pitch classes는 fallback으로 유지한다.
- final cadence landing에서 strict interval bound가 깨지지 않도록 최근 pitch blocking을 완화했다.

## 결과

Harness:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Summary:

| mode | strict | landing | max interval | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data_motif_contour_landing_repair` | 3/3 | 1.000 | 7 | 0.625 | 1.000 | 0.062 | 0.375 | 0.079 | 0.429 | 0.443 | 0.016 |
| `data_motif_rhythm_phrase_variation` | 3/3 | 1.000 | 4 | 0.684 | 0.958 | 0.079 | 0.384 | 0.091 | 0.385 | 0.437 | 0.016 |

Compared with Issue #130:

| metric | Issue #130 | Issue #134 |
|---|---:|---:|
| avg tension ratio | 0.358 | 0.437 |
| avg unique bar-position pattern ratio | 0.958 | 0.958 |
| avg IOI diversity ratio | 0.091 | 0.091 |
| avg most-common IOI ratio | 0.385 | 0.385 |
| max interval | 4 | 4 |
| duplicate note sequence count | 0 | 0 |
| objective MIDI flags | `{}` | `{}` |

## Candidate Snapshot

| candidate | notes | pitches | sync | bar-var | dur-rep | ioi-var | ioi-rep | tension | landing | max interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 63 | 28 | 0.667 | 1.000 | 0.381 | 0.097 | 0.371 | 0.413 | `guide` | 4 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 63 | 16 | 0.683 | 1.000 | 0.413 | 0.097 | 0.419 | 0.460 | `guide` | 4 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 64 | 20 | 0.703 | 0.875 | 0.359 | 0.079 | 0.365 | 0.438 | `guide` | 4 |

Objective MIDI review:

- candidate count: `6`
- objective flag counts: `{}`
- variation candidates: `clean`
- duplicate note sequences: `0`
- before/after max simultaneous notes for variation review MIDI: `1/1`
- large leap ratio: `0.000`
- unresolved large leap ratio: `0.000`
- repeated pitch interval ratio: `0.000-0.032`

Objective tension ratios for repaired variation candidates:

| candidate | chord-tone | tension | root |
|---|---:|---:|---:|
| `rank_1_sample_3` | 0.444 | 0.540 | 0.016 |
| `rank_2_sample_1` | 0.492 | 0.508 | 0.016 |
| `rank_3_sample_2` | 0.484 | 0.500 | 0.016 |

## 해석

Issue #134 preserves the rhythm/position guardrails while restoring tension color.

Positive signs:

- rank 1 no longer starts as an extreme high-register sketch; its first 16 notes sit mostly around D4-A#4 before moving.
- avg tension ratio is now close to the contour repair baseline.
- objective flags remain empty.
- all variation review MIDI is already monophonic before overlap-free export.

Remaining risks:

- rank 2 now has lower unique pitch count (`16`) and still forms a high-register arc.
- rank 3 remains a safe exercise-like line by proxy.
- no filled review notes exist for the repaired candidates yet.

## Decision

Issue #134 should move to review, not broad training.

Recommended next issue:

```text
Stage B phrase-shape tension repaired MIDI-note proxy review
```

Review target:

- whether rank 1 is now stronger than Issue #132 rank 1
- whether tension repair reduces `too_safe`
- whether high-register arc and exercise-like behavior remain blockers
- whether any candidate can become `keep` by proxy, or whether another phrase-shape repair is needed

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Commit 전 필수 quick harness도 실행한다.
