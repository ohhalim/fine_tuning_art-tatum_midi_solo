# Stage B Coverage-Aware A/B Sweep

작성일: 2026-05-20

## Issue

- Issue: #39
- Branch: `issue-39-stage-b-coverage-ab-sweep`
- Goal: plain constrained generation과 coverage-aware constrained generation을 같은 2-file Brad setup에서 비교한다.

## Why

Issue #37은 coverage-aware `POSITION` selection이 dead-air failure를 줄일 수 있음을 보였다.

하지만 #37은 단일 설정이었다.

이번 이슈에서는 다음을 비교한다.

- mode: `plain`, `coverage`
- note groups per bar: `4`, `6`, `8`
- top_k: `2`
- temperature: `0.9`
- samples per config: `3`

## Command

```bash
bash scripts/agent_harness.sh stage-b-coverage-ab-sweep
```

Generated outputs:

```text
outputs/stage_b_coverage_ab_sweep/harness_stage_b_coverage_ab_sweep/ab_sweep_report.json
outputs/stage_b_coverage_ab_sweep/harness_stage_b_coverage_ab_sweep/ab_sweep_report.md
```

Generated MIDI/checkpoint artifacts stay local and are not committed.

## Result

Summary:

| Metric | Value |
|---|---:|
| configs | 6 |
| plain configs | 3 |
| coverage configs | 3 |
| passed A/B sweep gate | true |
| best mode | coverage |
| best groups/bar | 8 |
| best strict valid samples | 3/3 |
| best onset coverage | 0.500 |
| best max longest sustained empty run | 1 |

Detailed rows:

| Mode | Groups/bar | Grammar | Basic valid | Strict valid | Onset | Sustained | Span | Max empty | Avg dead-air | Collapse warning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain | 4 | 3/3 | 0/3 | 0/3 | 0.167 | 0.417 | 0.740 | 11 | 0.811 | 0.000 |
| coverage | 4 | 3/3 | 3/3 | 3/3 | 0.250 | 0.427 | 0.813 | 6 | 0.429 | 0.000 |
| plain | 6 | 3/3 | 1/3 | 1/3 | 0.208 | 0.500 | 0.802 | 9 | 0.768 | 0.000 |
| coverage | 6 | 3/3 | 3/3 | 3/3 | 0.375 | 0.688 | 0.906 | 3 | 0.455 | 0.000 |
| plain | 8 | 3/3 | 2/3 | 2/3 | 0.240 | 0.604 | 0.875 | 6 | 0.710 | 0.333 |
| coverage | 8 | 3/3 | 3/3 | 3/3 | 0.500 | 0.865 | 0.938 | 1 | 0.467 | 0.000 |

## Interpretation

Coverage-aware constrained generation clearly improves temporal coverage in this harness.

Most important comparisons:

- plain g4 fails all strict samples because dead-air remains high.
- coverage g4 fixes the #35/#37 baseline failure.
- coverage g6 and g8 improve onset/sustained coverage further.
- coverage g8 gives the best temporal coverage, but chord-tone ratio is lower than g4.

This means density alone is not the final answer.

The next model-quality decision should consider:

- temporal coverage
- chord-tone ratio
- pitch diversity
- repeated pitch/position behavior
- whether the output still sounds like a playable solo line

## Decision

Do not jump straight to broad generic training yet.

The next issue should add a selection/ranking layer or quality report that chooses candidate MIDI based on multiple metrics instead of only strict pass/fail.

Recommended next issue:

- Stage B candidate ranking report
- choose best sample/config using temporal coverage, strict validity, chord-tone ratio, repetition, and density
- keep generation constrained for now
- only after this is stable, move toward generic jazz base training

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_coverage_ab_sweep tests.test_stage_b_sampling_sweep
./.venv/bin/python -m compileall scripts/run_stage_b_coverage_ab_sweep.py scripts/run_stage_b_sampling_sweep.py scripts/run_stage_b_generation_probe.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-coverage-ab-sweep
```
