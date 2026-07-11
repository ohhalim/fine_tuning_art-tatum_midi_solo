# Stage B Temporal Coverage Diagnostics

작성일: 2026-05-20

## Goal

Issue #35 explains why the Stage B 2-file Brad probe fails the MIDI review gate even though grammar and collapse checks are now stable.

The observed failure is dead-air:

- grammar gate: `3/3`
- collapse warning: `0/3`
- basic MIDI valid: `0/3`
- strict valid: `0/3`

This means the next bottleneck is not one-note collapse. It is temporal coverage.

## Added Diagnostics

Each generated sample now includes `temporal_coverage` in `report.json`.

Fields:

- `unique_onset_position_count`
- `onset_coverage_ratio`
- `sustained_coverage_ratio`
- `earliest_absolute_position`
- `latest_absolute_position`
- `position_span_steps`
- `position_span_ratio`
- `head_empty_steps`
- `tail_empty_steps`
- `longest_onset_empty_run_steps`
- `longest_sustained_empty_run_steps`
- `per_bar_unique_onset_positions`
- `per_bar_onset_coverage_ratio`

The probe summary aggregates:

- `avg_onset_coverage_ratio`
- `avg_sustained_coverage_ratio`
- `avg_position_span_ratio`
- `max_longest_sustained_empty_run_steps`

## Validation

Commands:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe
bash scripts/agent_harness.sh stage-b-2file-brad-probe
bash scripts/agent_harness.sh quick
```

Result:

- targeted Stage B generation probe tests: passed
- 2-file Brad probe harness: passed as an execution/reporting probe
- quick harness: passed

## Probe Result

Output:

```text
outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe
```

Summary:

| metric | value |
|---|---:|
| grammar gate | 3/3 |
| basic MIDI valid | 0/3 |
| strict valid | 0/3 |
| collapse warning | 0/3 |
| avg onset coverage ratio | 0.167 |
| avg sustained coverage ratio | 0.417 |
| avg position span ratio | 0.740 |
| max longest sustained empty run | 11 steps |

Per sample:

| sample | onset coverage | sustained coverage | position span | longest sustained empty run |
|---:|---:|---:|---:|---:|
| 1 | 0.156 | 0.344 | 0.844 | 11 |
| 2 | 0.188 | 0.469 | 0.625 | 9 |
| 3 | 0.156 | 0.438 | 0.750 | 5 |

## Interpretation

The generated phrases contain complete Stage B note groups and avoid the previous repeated position/pitch collapse.

However, onsets occupy only about one sixth of the available 2-bar 16th-note grid positions. Sustained coverage is higher, but long empty spans remain. This explains why phrase coverage can look moderate while dead-air still fails.

The model is choosing a small number of position buckets repeatedly enough to sound sparse, but not repeatedly enough to trigger the collapse gate.

## Decision

Do not move to generic jazz base training yet.

The next issue should test coverage-aware constrained generation:

- guide or constrain only `POSITION_*` selection
- keep pitch, duration, and velocity model-driven
- compare plain constrained generation against coverage-aware constrained generation
- require report-level improvement in dead-air and temporal coverage before scaling data

Next issue:

```text
Stage B coverage-aware constrained generation 추가
```
