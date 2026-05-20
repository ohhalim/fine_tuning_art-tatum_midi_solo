# Stage B 2-File Brad Generation Probe

작성일: 2026-05-20

## Goal

Issue #33 checks whether the Stage B generation path still produces reviewable MIDI after moving from a one-file tiny smoke to a Brad Mehldau 2-file setup.

This is still a local model-core probe, not a finished style model.

## Setup

Command:

```bash
bash scripts/agent_harness.sh stage-b-2file-brad-probe
```

Dataset:

- input: `./midi_dataset/midi/studio/Brad Mehldau`
- max files: `2`
- sequence format: `stage_b_v1`
- window bars: `2`
- window stride bars: `2`
- min target notes per window: `4`

Training/generation:

- epochs: `3`
- batch size: `8`
- max sequence: `96`
- generation mode: constrained
- note groups per bar: `4`
- sampling: `top_k=2`, temperature `0.9`
- samples: `3`
- overlap postprocess: enabled
- strict collapse gate: enabled

## Dataset Result

Output:

```text
outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe
```

Prepared records:

- input MIDI files: `2`
- total Stage B windows: `137`
- train samples: `123`
- val samples: `14`
- tokenized train: `123`
- tokenized val: `14`
- max token id: `544`
- vocab size: `547`

Training:

- epoch 1 val loss: `4.8525`
- epoch 2 val loss: `4.1989`
- epoch 3 val loss: `4.0892`

The dataset and model-vocab path are valid.

## Generation Result

| metric | value |
|---|---:|
| generated samples | 3 |
| grammar gate | 3/3 |
| basic MIDI valid | 0/3 |
| strict valid | 0/3 |
| collapse warning | 0/3 |
| avg repeated position/pitch pair ratio | 0.375 |
| avg postprocess removal ratio | 0.208 |

Failure reasons:

- sample 1: `dead-air ratio too high: 0.800 >= 0.800`
- sample 2: `dead-air ratio too high: 0.833 >= 0.800`
- sample 3: `dead-air ratio too high: 0.800 >= 0.800`

## Interpretation

This is not the same failure as the earlier one-note collapse.

What improved:

- Stage B grammar is stable in the 2-file probe.
- No sample triggered collapse warning.
- Unique pitch/position/pair counts are no longer the immediate blocker.
- Postprocess is not deleting most notes.

What failed:

- All samples still fail the MIDI review gate.
- The failure is temporal coverage/dead-air, not grammar or collapse.
- Generated positions are too clustered or do not cover enough of the 2-bar phrase.

## Decision

Do not move to generic jazz base training yet.

The next bottleneck to isolate is temporal coverage:

- per-bar occupied position count
- earliest/latest generated position
- longest empty span
- MIDI phrase coverage
- whether dead-air comes from position sampling, duration choice, or postprocess removal

## Next Issue

```text
Stage B temporal coverage diagnostics 추가
```

The next issue should add explicit coverage diagnostics and then test whether a coverage-aware constrained generation mode can recover at least one basic/strict valid sample without pretending that heavy postprocess is model quality.
