# Stage B Rhythm Variation Phrase-Vocabulary Diversity Repair

작성일: 2026-05-25

## 목적

Issue #130은 Issue #128 proxy review에서 남은 `too_stiff=6`, `keep=0` 병목을 generation rule 쪽에서 한 번 더 좁힌 작업이다.

목표:

- Issue #126 timing-grid repair guardrail을 유지한다.
- duplicate note sequence count를 `0`으로 유지한다.
- objective MIDI flags를 `{}`로 유지한다.
- dominant IOI repetition을 다시 키우지 않는다.
- bar-position, IOI, phrase contour vocabulary를 넓힌다.

## 변경

`data_motif_rhythm_phrase_variation`에 다음을 추가했다.

- 8-step slot split cycle로 bar-position pattern 반복을 줄였다.
- phrase position anti-repeat pattern을 3개에서 8개로 늘렸다.
- seed/bar/motif 기반 local position offset을 추가했다.
- duration variation index를 더 긴 cycle로 바꿨다.
- motif tail duration이 다음 motif를 덮지 않도록 next slot boundary로 제한했다.
- `phrase_vocabulary_contour_delta()`를 추가해 call/response motif에서 contour delta를 반전하거나 bias한다.
- pitch-cell selection에도 bar/motif shift를 넣어 같은 contour/pitch role 반복을 줄였다.

## 결과

Harness:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Summary:

| mode | strict | landing | max interval | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data_motif_contour_landing_repair` | 3/3 | 1.000 | 7 | 0.625 | 1.000 | 0.062 | 0.375 | 0.079 | 0.429 | 0.443 | 0.016 |
| `data_motif_rhythm_phrase_variation` | 3/3 | 1.000 | 4 | 0.684 | 0.958 | 0.079 | 0.384 | 0.091 | 0.385 | 0.358 | 0.005 |

Compared with Issue #126 timing-grid repair:

| metric | Issue #126 | Issue #130 |
|---|---:|---:|
| avg unique bar-position pattern ratio | 0.583 | 0.958 |
| avg IOI diversity ratio | 0.070 | 0.091 |
| avg most-common IOI ratio | 0.412 | 0.385 |
| avg most-common duration ratio | 0.416 | 0.384 |
| max interval | 4 | 4 |
| duplicate note sequence count | 0 | 0 |
| objective MIDI flags | `{}` | `{}` |

The duration diversity ratio is `0.079`, below the Issue #126 `0.084` ratio, but every variation candidate now has `5` unique objective MIDI duration values and lower dominant duration repetition.

## Candidate Snapshot

| candidate | notes | pitches | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | landing | max interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 63 | 28 | 0.667 | 1.000 | 0.079 | 0.381 | 0.097 | 0.371 | `guide` | 4 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 63 | 22 | 0.683 | 1.000 | 0.079 | 0.413 | 0.097 | 0.419 | `guide` | 4 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 64 | 23 | 0.703 | 0.875 | 0.078 | 0.359 | 0.079 | 0.365 | `guide` | 4 |

Objective MIDI review:

- candidate count: `6`
- objective flag counts: `{}`
- variation candidates: `clean`
- duplicate note sequences: `0`
- before/after max simultaneous notes for variation review MIDI: `1/1`
- large leap ratio: `0.000`
- unresolved large leap ratio: `0.000`
- repeated pitch interval ratio: `0.000-0.016`

## 해석

Issue #130 improves the objective surface that Issue #128 requested:

- bar-position vocabulary no longer repeats on the previous short cycle.
- IOI diversity recovers above the contour baseline and Issue #126 value.
- dominant IOI repetition drops below both Issue #126 and the contour baseline.
- overlap-free review export no longer needs trimming for variation candidates.
- objective-clean and duplicate-free guardrails remain intact.

This is still not a musical success claim:

- no filled listening/proxy review exists for these repaired candidates yet.
- tension ratio is lower than the contour repair baseline.
- duration diversity ratio is only modestly above the contour repair baseline.

## Decision

Issue #130 should move to review, not broad training.

Recommended next issue:

```text
Stage B phrase-vocabulary repaired rhythm MIDI-note proxy review
```

Review target:

- whether bar-position and IOI vocabulary repair reduces `too_stiff`
- whether lower tension ratio makes the line too safe
- whether rank 1 sample 3 is a better review candidate than the previous timing-grid repaired rank 1
- whether another phrase-vocabulary repair is needed before training scope expands

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Commit 전 필수 quick harness도 실행한다.
