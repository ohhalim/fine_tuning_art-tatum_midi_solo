# Stage B Pitch Mode Compare

작성일: 2026-05-21

## Context

Manual piano-roll review에서 4-bar 후보는 one-note/two-note failure는 아니고, 멜로디처럼 보이지만 너무 안전한 chord-tone line처럼 들린다는 피드백이 있었다.

Issue #53에서 root bias를 측정한 결과:

- average root tone ratio: about `0.271`
- top candidate root tone ratio: about `0.219`
- tension ratio: `0.000`

따라서 다음 질문은 "근음만 치는 collapse인가?"가 아니라:

> `tones_tensions`를 허용하면 root/chord-tone stiffness가 줄고 더 solo-line처럼 들을 만한 후보가 나오는가?

## Scope

Issue #55:

- 같은 tiny checkpoint로 `tones`와 `tones_tensions`를 비교한다.
- 기존 4-bar longer phrase 조건을 유지한다.
- 각 mode별 root/tension/chord-tone ratio를 report에 남긴다.
- 각 mode별 manual review MIDI를 따로 export한다.

Non-goals:

- broad training
- generic jazz base training
- pretrained model integration
- DAW/API/backend work

## Implementation

Added:

- `scripts/run_stage_b_pitch_mode_compare.py`
- `tests/test_stage_b_pitch_mode_compare.py`
- `scripts/agent_harness.sh stage-b-pitch-mode-compare`
- `scripts/run_stage_b_sampling_sweep.py` row summary fields:
  - `avg_root_tone_ratio`
  - `avg_tension_ratio`

Harness:

```bash
bash scripts/agent_harness.sh stage-b-pitch-mode-compare
```

The harness:

1. trains/prepares once for the first pitch mode
2. reuses the same checkpoint for the second pitch mode
3. writes a combined comparison report
4. exports review MIDI under mode-specific directories

## Local Result

Run:

```text
RUN_ID=harness_stage_b_pitch_mode_compare
```

Report:

```text
outputs/stage_b_pitch_mode_compare/harness_stage_b_pitch_mode_compare/pitch_mode_compare_report.json
outputs/stage_b_pitch_mode_compare/harness_stage_b_pitch_mode_compare/pitch_mode_compare_report.md
```

Review MIDI:

```text
outputs/stage_b_review_candidates/harness_stage_b_pitch_mode_compare/tones/midi/
outputs/stage_b_review_candidates/harness_stage_b_pitch_mode_compare/tones_tensions/midi/
```

Summary:

| pitch mode | strict samples | chord | root | tension | onset | sustained | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tones` | 3/3 | 0.927 | 0.271 | 0.000 | 0.500 | 0.682 | 0.000 |
| `tones_tensions` | 3/3 | 0.667 | 0.135 | 0.313 | 0.500 | 0.667 | 0.000 |

Deltas, `tones_tensions - tones`:

- root tone ratio: `-0.135`
- tension ratio: `+0.313`

## Interpretation

`tones_tensions` did what it was supposed to do numerically:

- root-tone usage dropped by about half
- tension usage appeared
- strict MIDI validity stayed at 3/3
- temporal coverage stayed comparable

But it is not automatically better musically:

- chord-tone ratio drops because tensions are counted separately from chord tones
- top review candidates still carry `high_repeated_pitch_ratio`
- `tones_tensions` candidates also show stronger dominant-pitch risk than the best `tones` candidate

Current diagnosis:

> `tones` sounds safe and inside, while `tones_tensions` is harmonically less stiff but still mechanically constrained by the current pitch-selection policy.

## Review Instruction

Manual listening should compare mode pairs, not isolated files.

Listen in this order:

1. `outputs/stage_b_review_candidates/harness_stage_b_pitch_mode_compare/tones/midi/rank_01_coverage_chord_g8_s3.mid`
2. `outputs/stage_b_review_candidates/harness_stage_b_pitch_mode_compare/tones_tensions/midi/rank_01_coverage_chord_g8_s3.mid`
3. compare whether the second file sounds more like a phrase or only more chromatic/tension-heavy

Review question:

- Is `tones_tensions` more solo-like, or just less inside?
- Does the line still sound like it is selecting from a small pitch set?
- Is repeated-pitch feel worse than the previous `tones` candidate?

## Decision

Do not jump to broad training yet.

The next useful step is phrase-shape control:

- either add passing/approach pitch policy instead of unrestricted tensions
- or add motif/contour constraints that reduce repeated pitch-class cycling
- then compare against this `tones_tensions` baseline

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_pitch_mode_compare tests.test_stage_b_generation_probe tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/run_stage_b_pitch_mode_compare.py scripts/run_stage_b_sampling_sweep.py scripts/agent_harness.sh tests/test_stage_b_pitch_mode_compare.py
bash scripts/agent_harness.sh stage-b-pitch-mode-compare
```
