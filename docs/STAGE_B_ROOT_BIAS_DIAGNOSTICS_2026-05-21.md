# Stage B Root Bias Diagnostics

작성일: 2026-05-21

## Issue

- Issue: #53
- Branch: `issue-53-stage-b-root-bias-diagnostics`
- Goal: "근음을 계속 치는 느낌"을 root tone ratio와 tension ratio로 분리해 진단한다.

## Why

Manual review of the 4-bar candidates changed the diagnosis.

The output is no longer obviously broken.

Current listening feedback:

- melody-like enough to review
- not a one-note or two-note failure
- not adjacent same-note collapse
- but it feels like the line keeps leaning on roots

Before changing the generator again, this needs a numeric answer.

## Implementation

Added `pitch_roles` diagnostics to Stage B sample reports:

- `root_tone_count`
- `root_tone_ratio`
- `chord_tone_count`
- `chord_tone_ratio`
- `non_root_chord_tone_count`
- `non_root_chord_tone_ratio`
- `tension_count`
- `tension_ratio`
- `non_chord_tone_count`
- `non_chord_tone_ratio`
- `per_bar_root_tone_ratio`

Updated review export:

- root ratio column
- tension ratio column
- root-bias risk flag support

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
| average root tone ratio | 0.271 |
| top candidate root tone ratio | 0.219 |
| chord tone ratio | 0.938-1.000 |
| tension ratio | 0.000 |
| adjacent repeated pitch ratio | 0.000 |
| average direction change ratio | 0.689 |

Top exported candidates:

| rank | root | tension | risk |
|---:|---:|---:|---|
| 1 | 0.219 | 0.000 | `high_repeated_pitch_ratio` |
| 2 | 0.312 | 0.000 | `high_repeated_pitch_ratio`, `high_dominant_pitch_ratio` |
| 3 | 0.281 | 0.000 | `high_repeated_pitch_ratio` |

## Interpretation

The output is not pure root collapse.

The stronger diagnosis is:

> current `coverage_chord` output is a safe chord-tone-only line with no tensions.

So the listener can reasonably hear it as root-heavy or too inside, even when the actual root ratio is not extreme.

The current pitch constraint uses:

```text
--chord_pitch_mode tones
```

That means allowed pitches are only chord tones. Tensions are not available during constrained pitch selection.

## Decision Boundary

Do not start broad training only because the output is melody-like.

Next technical comparison should be:

```text
tones vs tones_tensions
```

Expected question:

- Does `tones_tensions` reduce root/chord-tone stiffness?
- Does it preserve enough chord compatibility?
- Does it create more melodic color without falling back into random non-chord pitches?

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/run_stage_b_generation_probe.py scripts/export_stage_b_review_candidates.py tests/test_stage_b_generation_probe.py tests/test_stage_b_review_export.py
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
bash scripts/agent_harness.sh quick
```
