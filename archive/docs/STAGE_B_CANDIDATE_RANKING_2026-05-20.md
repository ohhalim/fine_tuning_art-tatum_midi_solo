# Stage B Candidate Ranking

작성일: 2026-05-20

## Issue

- Issue: #41
- Branch: `issue-41-stage-b-candidate-ranking`
- Goal: Stage B generated MIDI candidates를 strict pass/fail 하나가 아니라 여러 지표로 ranking한다.

## Why

Issue #39에서 coverage-aware constrained generation은 plain constrained generation보다 temporal coverage를 안정적으로 개선했다.

하지만 tradeoff가 있었다.

- coverage groups/bar `8`은 temporal coverage가 가장 좋다.
- coverage groups/bar `4`는 chord-tone ratio가 더 높을 수 있다.
- strict gate 통과만으로 어떤 MIDI를 먼저 들어야 하는지 결정하기 어렵다.

따라서 candidate ranking report가 필요하다.

## Implementation

Added:

- `scripts/rank_stage_b_candidates.py`
- `tests/test_stage_b_candidate_ranking.py`
- `bash scripts/agent_harness.sh stage-b-candidate-ranking`

The ranking script reads an A/B sweep report, then loads each config-level `report.json` and ranks sample-level MIDI candidates.

Inputs:

```text
outputs/stage_b_coverage_ab_sweep/<run_id>/ab_sweep_report.json
```

Outputs:

```text
outputs/stage_b_candidate_ranking/<run_id>/candidate_rank_report.json
outputs/stage_b_candidate_ranking/<run_id>/candidate_rank_report.md
```

Generated MIDI artifacts remain local and are not committed.

## Scoring

The score is a review-prioritization heuristic.

It rewards:

- strict validity
- basic validity
- grammar validity
- onset coverage
- sustained coverage
- position span
- chord-tone ratio
- pitch diversity

It penalizes:

- dead-air ratio
- repetition score
- postprocess removal ratio
- collapse warning

This score is not a musical-quality claim.

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-candidate-ranking
```

Top candidate:

| Field | Value |
|---|---:|
| mode | coverage |
| groups/bar | 8 |
| sample index | 1 |
| score | 91.080 |
| strict valid | true |
| note count | 16 |
| unique pitches | 3 |
| onset coverage | 0.500 |
| sustained coverage | 0.906 |
| position span | 0.938 |
| longest sustained empty run | 1 |
| dead-air ratio | 0.467 |
| repetition score | 0.154 |
| chord-tone ratio | 0.313 |

Top candidate MIDI path:

```text
outputs/stage_b_coverage_ab_sweep/harness_stage_b_candidate_ranking_ab_sweep_coverage_g8_k2_t0p9/samples/stage_b_sample_1.mid
```

Top ranking pattern:

- ranks 1-3: coverage groups/bar `8`
- ranks 4-6: coverage groups/bar `6`
- ranks 7-9: coverage groups/bar `4`
- lower ranks include plain configs that passed strict gate but had worse dead-air/postprocess behavior

## Decision

The next review step should be listening/piano-roll inspection of top ranked candidates.

Do not start broad generic jazz training yet unless the top ranked candidates are at least structurally acceptable by ear and piano roll.

If the top candidates still sound harmonically weak, the next issue should be chord-aware pitch candidate filtering or chord-tone/tension-aware ranking.

## Follow-up Correction

Issue #43 performed that piano-roll review and found the top ranked candidate was not a usable solo-line candidate.

The original ranking over-weighted temporal coverage and did not penalize repeated pitch, repeated bar templates, or low per-bar chord-tone coverage strongly enough.

Use `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md` as the current interpretation of this result.

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_candidate_ranking
./.venv/bin/python -m compileall scripts/rank_stage_b_candidates.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-candidate-ranking
```
