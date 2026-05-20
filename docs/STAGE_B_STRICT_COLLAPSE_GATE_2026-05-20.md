# Stage B Strict Collapse Gate

작성일: 2026-05-20

## Goal

Issue #31 adds a stricter collapse-aware review gate on top of the existing Stage B MIDI review gate.

The purpose is to stop treating a generated MIDI as strong progress when it only passes after heavy postprocess or when token generation collapses into repeated position/pitch events.

## Gate Layers

The Stage B probe now reports three related gates:

- grammar gate: generated tokens form complete Stage B note groups
- basic MIDI gate: decoded MIDI passes the existing metrics validation
- strict collapse gate: the sample also has enough position/pitch diversity and low postprocess damage

The sampling sweep reports both:

- `passed_basic_sweep_gate`
- `passed_strict_sweep_gate`

`passed_sweep_gate` now follows the stricter gate.

## Strict Defaults

Per-sample strict gate defaults:

- minimum unique pitches: `3`
- minimum unique positions: `3`
- minimum unique position/pitch pairs: `4`
- max repeated position/pitch pair ratio: `0.49`
- max postprocess removal ratio: `0.49`

Per-sweep strict default:

- max collapse warning sample rate: `0.34`

These numbers are intentionally small because the current probe only generates three samples from a tiny one-file checkpoint. They are not final musical quality thresholds.

## Local Validation

Commands:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe tests.test_stage_b_sampling_sweep
bash scripts/agent_harness.sh stage-b-collapse-sweep
bash scripts/agent_harness.sh quick
```

Result:

- targeted unit tests: passed
- stage-b-collapse-sweep: passed
- quick harness: passed

## Sweep Result

Output:

```text
outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep
```

Summary:

| top_k | temperature | grammar | basic valid | strict valid | collapse warning | strict pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 0.9 | 3/3 | 0/3 | 0/3 | 3/3 | false |
| 2 | 0.9 | 3/3 | 1/3 | 1/3 | 1/3 | true |

Best config:

```text
top_k=2, temperature=0.9
```

## Interpretation

`top_k=1` is now clearly rejected by both the basic MIDI gate and the strict collapse gate.

`top_k=2` still keeps one reviewable candidate under the stricter gate. This is not enough to claim musical quality, but it is enough to move from one-file tiny smoke to a 2-file Brad Stage B probe.

The current bottleneck remains note distribution collapse, not Stage B grammar.

## Next Step

Next issue:

```text
Stage B 2-file Brad generation probe 추가
```

That issue should verify whether the one-file result survives a slightly broader Brad train/val setup before any generic jazz base training is started.
