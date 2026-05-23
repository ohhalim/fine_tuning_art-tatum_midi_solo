# Stage B Objective Clean Review Package

작성일: 2026-05-23

## Purpose

Issue #109는 Stage B data-motif phrase recovery review 결과에서 objective metric 기준으로 clean 후보만 따로 묶는 단계다.

이 단계의 목적은 새 generation rule을 추가하는 것이 아니다. 이전 probe에서 이미 만들어진 후보 중, 최소한 다음 조건을 만족하는 후보만 listening review 대상으로 줄인다.

- objective bucket: `clean`
- objective flags: none
- mode: `data_motif_phrase_recovery`
- review MIDI와 context MIDI 경로를 같이 제공

이렇게 해야 사람이 이상한 MIDI까지 모두 듣지 않고, 지금 가장 리뷰 가치가 높은 후보만 확인할 수 있다.

## Implementation

Added:

- `scripts/build_clean_review_package.py`
- `tests/test_clean_review_package.py`
- `bash scripts/agent_harness.sh stage-b-clean-review-package`

The package builder reads:

- review manifest
- objective MIDI note review report

Then it writes:

- `clean_review_package.json`
- `clean_review_package.md`
- optional copied solo MIDI files
- optional copied context MIDI files

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-clean-review-package
```

Output package:

```text
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md
```

Result:

- candidate count: `3`
- allowed mode: `data_motif_phrase_recovery`
- all selected candidates have no objective flags

Selected candidates:

| candidate | notes | unique pitches | unresolved large leap | chord tone | tension |
|---|---:|---:|---:|---:|---:|
| `data_motif_phrase_recovery_rank_1_sample_1` | 63 | 19 | 0.000 | 0.508 | 0.492 |
| `data_motif_phrase_recovery_rank_2_sample_2` | 63 | 23 | 0.000 | 0.524 | 0.476 |
| `data_motif_phrase_recovery_rank_3_sample_3` | 63 | 22 | 0.045 | 0.476 | 0.524 |

## Interpretation

This is not proof of jazz quality.

It only means the candidates no longer show the objective failures that previously blocked review:

- one-note/two-note failure
- chord-block review artifact
- overlap/polyphonic review artifact
- duration collapse
- scalar/chromatic objective flag
- unresolved large-leap objective flag

The next decision must come from listening/piano-roll review with chord context.

## Next Step

Use the context MIDI files from the clean package to answer:

- does the solo line fit the chord context?
- does the rhythm feel intentionally quantized, not timing-drifted?
- does the phrase have continuation and landing, not just isolated notes?
- does it sound like jazz vocabulary or only chord-tone/tension enumeration?

If these three clean candidates still sound beginner-like, the next generation work should not be another hand-written rule. It should move toward stronger data-derived phrase/cadence material or a model-side phrase continuation objective.
