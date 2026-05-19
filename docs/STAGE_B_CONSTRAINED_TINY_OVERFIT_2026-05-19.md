# Stage B Grammar-Constrained Tiny-Overfit: 2026-05-19

## Purpose

Issue #20 adds a stricter Stage B grammar smoke.

Issue #18 proved that Stage B generation can sample the full vocabulary and decode to MIDI, but the first unconstrained generated sample had zero decoded notes. The missing piece was not another broad training run. The next check was whether the model logits can be sampled under a simple Stage B note grammar:

```text
POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION
```

This issue separates two gates:

- grammar gate: generated tokens contain complete note groups and decode to at least one MIDI note
- full review gate: decoded MIDI passes the existing musical quality metrics

## Implementation

Code changes:

- `scripts/run_stage_b_generation_probe.py`
  - adds generated-token grammar analysis
  - adds `generation_mode=constrained`
  - constrains model sampling by token family:
    - `POSITION_*`
    - `VELOCITY_*`
    - `NOTE_PITCH_*`
    - `NOTE_DURATION_*`
  - reports `grammar_gate_passed`, `passed_grammar_gate`, and `passed_generation_gate` separately
  - adds `--require_note_groups` for smoke checks
- `scripts/agent_harness.sh`
  - adds `stage-b-constrained-probe`
- `tests/test_stage_b_generation_probe.py`
  - verifies grammar counting
  - verifies incomplete groups are reported
  - verifies constrained generation creates decodable notes

## Command

```bash
bash scripts/agent_harness.sh stage-b-constrained-probe
```

Equivalent direct command:

```bash
python scripts/run_stage_b_generation_probe.py \
  --run_id harness_stage_b_constrained_probe \
  --max_files 1 \
  --epochs 1 \
  --batch_size 8 \
  --max_sequence 96 \
  --num_samples 1 \
  --generation_mode constrained \
  --constrained_note_groups_per_bar 4 \
  --top_k 1 \
  --require_note_groups \
  --n_layers 1 \
  --num_heads 4 \
  --d_model 64 \
  --dim_feedforward 128 \
  --lora_r 4 \
  --lora_alpha 8
```

## Local Result

Output:

```text
outputs/stage_b_generation_probe/harness_stage_b_constrained_probe
```

Dataset/training:

| Metric | Value |
|---|---:|
| role samples | 70 |
| train records | 63 |
| val records | 7 |
| max token id | 544 |
| epoch 1 train loss | 6.2115 |
| epoch 1 val loss | 5.9441 |

Grammar result:

| Metric | Value |
|---|---:|
| generated samples | 1 |
| complete note groups | 8 |
| invalid grammar tokens | 0 |
| decoded note count | 8 |
| grammar gate sample count | 1 |
| passed grammar gate | true |

Full review gate:

| Metric | Value |
|---|---:|
| valid samples | 0 |
| passed generation gate | false |
| failure reason | too many simultaneous notes: 3 > 2 |
| note density | 2.204 |
| phrase coverage ratio | 0.937 |
| max simultaneous notes | 3 |

## Decision

Stage B can now produce complete note groups when generation is constrained by token family.

This is progress over Issue #18 because decoded MIDI now has real notes. It is still not a musically valid model result because the full review gate failed on overlapping notes.

## Next Step

Do not start broad jazz training yet.

Next issue should reduce overlap and repeated-position artifacts before increasing data size:

- prevent duplicate notes at the same bar/position/pitch
- optionally sort or de-overlap constrained generated notes before metrics
- add a `max_simultaneous_notes <= 2` constrained smoke gate
- only then try a longer tiny-overfit or a 2-file Stage B probe
