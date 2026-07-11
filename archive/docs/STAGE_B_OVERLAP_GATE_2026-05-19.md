# Stage B Overlap Gate: 2026-05-19

## Purpose

Issue #22 removes the next concrete blocker in Stage B constrained generation.

Issue #20 proved that constrained generation can produce complete Stage B note groups and decode them into MIDI notes. The remaining full-gate failure was overlap:

```text
too many simultaneous notes: 3 > 2
```

This issue adds a small postprocess step for constrained diagnostic output:

- remove duplicate notes at the same onset/pitch
- limit overlapping active notes to a configured maximum
- report before/after overlap metrics

This is still not broad jazz training. It is a narrow gate check for the Stage B local generation path.

## Implementation

Code changes:

- `scripts/run_stage_b_generation_probe.py`
  - adds `dedupe_and_limit_notes`
  - adds `postprocess_stage_b_midi`
  - adds `--postprocess_overlap`
  - adds `--max_simultaneous_notes`
  - records postprocess before/after note counts and max simultaneous notes
- `scripts/agent_harness.sh`
  - adds `stage-b-overlap-gate`
- `tests/test_stage_b_generation_probe.py`
  - verifies duplicate same-onset/same-pitch notes are removed
  - verifies max simultaneous notes are limited

## Command

```bash
bash scripts/agent_harness.sh stage-b-overlap-gate
```

Equivalent direct command:

```bash
python scripts/run_stage_b_generation_probe.py \
  --run_id harness_stage_b_overlap_gate \
  --max_files 1 \
  --epochs 1 \
  --batch_size 8 \
  --max_sequence 96 \
  --num_samples 1 \
  --generation_mode constrained \
  --constrained_note_groups_per_bar 4 \
  --postprocess_overlap \
  --max_simultaneous_notes 2 \
  --top_k 1 \
  --require_note_groups \
  --require_valid_sample \
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
outputs/stage_b_generation_probe/harness_stage_b_overlap_gate
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

Postprocess:

| Metric | Value |
|---|---:|
| before note count | 8 |
| after note count | 6 |
| removed notes | 2 |
| before max simultaneous notes | 3 |
| after max simultaneous notes | 2 |

Final metrics:

| Metric | Value |
|---|---:|
| complete note groups | 8 |
| decoded note count | 6 |
| unique pitch count | 5 |
| note density | 1.653 |
| phrase coverage ratio | 0.937 |
| max simultaneous notes | 2 |
| passed grammar gate | true |
| passed generation gate | true |

## Decision

This is the first Stage B constrained smoke that passes the local MIDI review gate.

The result is still diagnostic, not a claim of musical quality:

- generation is constrained by token family
- overlap postprocess is enabled
- the training run is only one epoch on one Brad file subset
- chord-tone ratio is low and not yet used as a hard gate

## Next Step

Do not jump directly to broad training.

The next issue should run a slightly stronger Stage B tiny-overfit probe:

- longer tiny training on the same short window set
- constrained generation with overlap postprocess
- multiple seeds/samples
- require all generated samples to pass grammar gate
- require at least one sample to pass full review gate without hiding failure cases
