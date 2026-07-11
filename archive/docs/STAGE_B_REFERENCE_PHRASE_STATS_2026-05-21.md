# Stage B Reference Phrase Statistics

작성일: 2026-05-21

## Context

Issue #59 improved the generated MIDI rhythm grammar:

- baseline approach grammar repeated a mechanical position/IOI template.
- `swing_motif_approach` increased syncopation and bar-to-bar position variation.

But this was still a hand-written rule.

The next question is:

> Is the generated phrase rhythm actually close to real jazz MIDI phrase windows, or just less bad than the previous baseline?

Issue #61 builds a reference statistics report from real Stage B phrase windows.

## Goal

The goal is not to train a broader model yet.

Goal:

- prepare real jazz MIDI phrase windows with Stage B tokenization
- compute phrase-level reference statistics
- compare generated phrase grammar outputs against those reference means
- use data-derived numbers before adding more hand-written rules

## Implementation

Added:

- `scripts/run_stage_b_reference_stats.py`
- `tests/test_stage_b_reference_stats.py`
- `bash scripts/agent_harness.sh stage-b-reference-stats`

The script:

1. runs `prepare_role_dataset.py` with `sequence_format=stage_b_v1`
2. extracts 8-bar phrase windows
3. reads tokenized `.npy` records
4. computes reference metrics from real MIDI windows
5. optionally compares generated reports against the reference mean

Tracked reference metrics:

- `note_group_count`
- `unique_pitch_count`
- `pitch_span`
- `repeated_pitch_ratio`
- `syncopated_onset_ratio`
- `unique_bar_position_pattern_ratio`
- `duration_diversity_ratio`
- `most_common_duration_ratio`
- `ioi_diversity_ratio`
- `most_common_ioi_ratio`
- `direction_change_ratio`
- `stepwise_motion_ratio`
- `leap_motion_ratio`
- `onset_coverage_ratio`
- `sustained_coverage_ratio`

## Local Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-reference-stats
```

Report:

```text
outputs/stage_b_reference_stats/harness_stage_b_reference_stats/reference_stats_report.json
outputs/stage_b_reference_stats/harness_stage_b_reference_stats/reference_stats_report.md
```

Setup:

- input dir: `./midi_dataset/midi/studio`
- max files: `4`
- window bars: `8`
- stride bars: `4`
- min target notes: `16`
- analyzed records: `57`

Reference summary:

| metric | mean | p50 |
|---|---:|---:|
| `note_group_count` | 32.649 | 27.000 |
| `unique_pitch_count` | 11.000 | 10.000 |
| `pitch_span` | 19.825 | 14.000 |
| `repeated_pitch_ratio` | 0.642 | 0.647 |
| `syncopated_onset_ratio` | 0.736 | 0.750 |
| `unique_bar_position_pattern_ratio` | 0.996 | 1.000 |
| `duration_diversity_ratio` | 0.379 | 0.364 |
| `most_common_duration_ratio` | 0.260 | 0.250 |
| `ioi_diversity_ratio` | 0.341 | 0.346 |
| `most_common_ioi_ratio` | 0.339 | 0.333 |
| `direction_change_ratio` | 0.644 | 0.667 |
| `stepwise_motion_ratio` | 0.346 | 0.375 |
| `leap_motion_ratio` | 0.348 | 0.318 |
| `onset_coverage_ratio` | 0.189 | 0.156 |
| `sustained_coverage_ratio` | 0.711 | 0.750 |

Generated comparison against Issue #59:

| grammar | sync delta | bar-var delta | dur-var delta | dur-rep delta | ioi-var delta | ioi-rep delta |
|---|---:|---:|---:|---:|---:|---:|
| `approach_baseline` | -0.236 | -0.871 | -0.286 | +0.292 | -0.309 | +0.169 |
| `swing_motif_approach` | +0.014 | -0.496 | -0.307 | +0.120 | -0.278 | +0.137 |

## Interpretation

This gives a clearer answer than manual guesswork.

What improved:

- `swing_motif_approach` syncopation is near the reference mean.
- `swing_motif_approach` reduced duration repetition compared with baseline.
- generated MIDI is no longer one-note/two-note/chord-block failure.

What is still weak:

- real MIDI windows almost never repeat the same bar-position pattern.
- generated bar-position pattern variation is still far below reference.
- generated duration diversity is far below reference.
- generated IOI diversity is far below reference.
- generated most-common duration/IOI ratios are still too high.

Current diagnosis:

> The next problem is not simply "add more swing." The generated phrase needs more data-like rhythmic/motif variation across bars.

## Decision

Do not move to broad training only because Issue #59 sounds better than baseline.

Next useful options:

1. derive rhythm/motif templates from real Stage B windows instead of hand-writing them
2. add phrase-ending/cadence constraints and compare against reference windows
3. add motif-level pitch cells with interval movement statistics

Recommended next issue:

> Stage B data-derived phrase motif template extraction.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_reference_stats
./.venv/bin/python -m compileall scripts/run_stage_b_reference_stats.py tests/test_stage_b_reference_stats.py scripts/agent_harness.sh
bash scripts/agent_harness.sh stage-b-reference-stats
```
