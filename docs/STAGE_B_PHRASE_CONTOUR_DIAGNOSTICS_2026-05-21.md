# Stage B Phrase Contour Diagnostics

작성일: 2026-05-21

## Issue

- Issue: #51
- Branch: `issue-51-stage-b-phrase-contour-diagnostics`
- Goal: 4-bar `coverage_chord` 후보의 repeated-pitch risk를 더 구체적으로 설명한다.

## Why

Issue #49는 2-bar 후보가 너무 짧고 미완성처럼 보이는 문제를 4-bar probe로 해결했다.

하지만 새 후보의 repeated pitch ratio가 약 `0.719`로 높았다.

이 값만 보면 두 가지를 구분할 수 없다.

- 같은 음을 바로 이어서 반복하는 collapse
- 제한된 chord-tone pitch set을 많이 재사용하는 motif-like pattern

이 둘은 다르게 다뤄야 한다.

Adjacent same-note collapse라면 generation constraint를 고쳐야 한다.
Pitch-set reuse라면 실제 청취에서 motif로 들리는지 먼저 판단해야 한다.

## Implementation

Added:

- `analyze_stage_b_phrase_contour()`
- `phrase_contour` block in Stage B sample reports
- `avg_adjacent_repeated_pitch_ratio`
- `avg_direction_change_ratio`
- `max_longest_same_pitch_run`
- review export `risk_flags`

The review exporter now surfaces risk without dropping candidates.

Example flags:

- `high_repeated_pitch_ratio`
- `high_dominant_pitch_ratio`
- `adjacent_pitch_repetition`
- `contour:long_same_pitch_run`
- `contour:low_direction_change`
- `contour:low_interval_variety`

## Diagnostics

Per sample contour fields:

| Field | Meaning |
|---|---|
| `adjacent_repeated_pitch_ratio` | Consecutive same-pitch interval ratio |
| `direction_change_ratio` | How often melodic direction changes after removing zero intervals |
| `longest_same_pitch_run` | Longest adjacent same-pitch run |
| `unique_interval_count` | Number of distinct non-zero melodic intervals |
| `pitch_span` | Max pitch minus min pitch |
| `stepwise_motion_ratio` | Non-zero intervals within 1-2 semitones |
| `leap_motion_ratio` | Non-zero intervals at 5+ semitones |

## Latest Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```

Result:

| Metric | Value |
|---|---:|
| generated samples | 3 |
| strict valid samples | 3 |
| repeated pitch ratio | 0.719 |
| adjacent repeated pitch ratio | 0.000 |
| avg direction change ratio | 0.689 |
| max longest same pitch run | 1 |
| collapse warning samples | 0 |

Review export now marks the top candidates with risk flags:

| candidate | risk |
|---|---|
| `rank_01_coverage_chord_g8_s3.mid` | `high_repeated_pitch_ratio` |
| `rank_02_coverage_chord_g8_s2.mid` | `high_repeated_pitch_ratio`, `high_dominant_pitch_ratio` |
| `rank_03_coverage_chord_g8_s1.mid` | `high_repeated_pitch_ratio` |

## Interpretation

This is not adjacent same-note collapse.

The model is not producing `C C C C` style adjacent repeated notes in the longer phrase probe.

The remaining issue is narrower:

- the pitch set is still constrained
- chord-tone ratio is high
- direction changes are frequent
- adjacent repeated pitch is zero
- but the same few chord-tone pitches are reused across the 4 bars

So the next review question is:

> Does the repeated pitch reuse sound like motif/inside playing, or does it sound like constrained pitch cycling?

## Decision Boundary

If the 4-bar candidates sound musical enough:

- proceed to generic jazz base training design
- keep the current contour diagnostics as review metadata

If they sound mechanical:

- do not start broad training yet
- consider motif-level generation controls
- consider a stronger pretrained symbolic MIDI base
- consider allowing controlled non-chord tension tones instead of chord tones only

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/run_stage_b_generation_probe.py scripts/export_stage_b_review_candidates.py tests/test_stage_b_generation_probe.py tests/test_stage_b_review_export.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```
