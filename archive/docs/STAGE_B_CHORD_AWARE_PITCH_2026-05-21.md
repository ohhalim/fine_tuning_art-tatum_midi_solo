# Stage B Chord-Aware Pitch Constraint

작성일: 2026-05-21

## Issue

- Issue: #45
- Branch: `issue-45-stage-b-chord-aware-pitch`
- Goal: Stage B constrained generation에서 `NOTE_PITCH` 후보군을 현재 bar chord에 맞게 제한해 harmonic/repetition gate를 통과하는 candidate를 만든다.

## Why

Issue #43은 기존 Stage B ranking이 나쁜 MIDI를 좋은 후보로 올리던 문제를 수정했다.

그 결과는 엄격했다.

- candidates: `18`
- strict candidates: `12`
- viable unflagged candidates: `0`
- flagged candidates: `18`

즉, temporal coverage와 strict MIDI gate만으로는 아직 reviewable solo-line candidate를 고를 수 없었다.

Issue #45는 ranking을 더 꾸미지 않고 generation-side pitch selection을 고친다.

## Implementation

Added to `scripts/run_stage_b_generation_probe.py`:

- `--chord_aware_pitches`
- `--chord_pitch_mode tones|tones_tensions`
- `--chord_pitch_repeat_window`

Generation behavior:

- `POSITION`은 기존 coverage-aware option을 유지할 수 있다.
- `VELOCITY`와 `DURATION`은 기존 constrained family sampling을 유지한다.
- `NOTE_PITCH` family에서만 current bar chord에 맞는 pitch token 후보군을 만든다.
- `tones` mode는 chord tones만 허용한다.
- `tones_tensions` mode는 chord tones plus limited tensions를 허용한다.
- recent exact pitch repeat window로 직전 pitch 반복을 줄인다.

Added to sweep/harness:

- `coverage_chord` mode
- `bash scripts/agent_harness.sh stage-b-chord-aware-probe`

The harness runs:

```text
plain
coverage
coverage_chord
```

Then it runs candidate ranking over the generated MIDI candidates.

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-chord-aware-probe
```

Ranking summary:

| Metric | Value |
|---|---:|
| candidates | 27 |
| valid candidates | 21 |
| strict candidates | 21 |
| viable unflagged candidates | 9 |
| flagged candidates | 18 |

Best candidate:

| Field | Value |
|---|---:|
| mode | coverage_chord |
| groups/bar | 4 |
| sample index | 2 |
| score | 96.6964 |
| note count | 8 |
| unique pitch count | 6 |
| chord-tone ratio | 0.750 |
| bar chord-tone ratio | 0.875 |
| min bar chord-tone ratio | 0.800 |
| dominant pitch ratio | 0.375 |
| repeated pitch ratio | 0.250 |
| onset coverage | 0.250 |
| sustained coverage | 0.531 |

Top candidate MIDI:

```text
outputs/stage_b_coverage_ab_sweep/harness_stage_b_chord_aware_probe_ab_sweep_coverage_chord_g4_k2_t0p9/samples/stage_b_sample_2.mid
```

## Comparison

Coverage-only vs coverage+chord-aware:

| groups/bar | mode | strict | avg chord-tone | avg repeated position/pitch | onset | sustained | max empty |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | coverage | 3 | 0.417 | 0.250 | 0.250 | 0.427 | 6 |
| 4 | coverage_chord | 3 | 0.708 | 0.042 | 0.250 | 0.438 | 5 |
| 6 | coverage | 3 | 0.306 | 0.194 | 0.375 | 0.688 | 3 |
| 6 | coverage_chord | 3 | 0.639 | 0.000 | 0.375 | 0.667 | 3 |
| 8 | coverage | 3 | 0.229 | 0.167 | 0.500 | 0.865 | 1 |
| 8 | coverage_chord | 3 | 0.646 | 0.021 | 0.500 | 0.844 | 2 |

## Decision

This is the first Stage B probe where ranking finds unflagged reviewable candidates.

This still does not prove:

- Brad style learning
- generic jazz base quality
- unconstrained generation quality
- live improvisation readiness

It proves a narrower point:

> Coverage-aware timing plus chord-aware pitch constraints can produce generated MIDI candidates that pass the current harmonic/repetition review flags.

## Next Step

Manual listening and piano-roll review is now required.

Listen to the top `coverage_chord` candidates and check:

- solo-line shape
- phrase contour
- over-mechanical rhythm
- excessive high-register bias
- whether chord-tone correctness sounds too constrained or usable

If the candidates are acceptable by ear, the next issue can design the generic jazz base training probe.

If they are still mechanical, the next issue should target rhythm/motif-level behavior before broad training.

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe
./.venv/bin/python -m unittest tests.test_stage_b_coverage_ab_sweep tests.test_stage_b_generation_probe
./.venv/bin/python -m compileall scripts/run_stage_b_generation_probe.py tests/test_stage_b_generation_probe.py
./.venv/bin/python -m compileall scripts/run_stage_b_coverage_ab_sweep.py scripts/run_stage_b_sampling_sweep.py scripts/run_stage_b_generation_probe.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-chord-aware-probe
```
