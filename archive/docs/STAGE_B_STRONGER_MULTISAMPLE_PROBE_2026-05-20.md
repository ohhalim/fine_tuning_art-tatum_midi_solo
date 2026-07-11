# Stage B Stronger Multi-Sample Probe: 2026-05-20

## Purpose

Issue #24 strengthens the previous Stage B overlap gate.

Issue #22 proved that one constrained/postprocessed Stage B sample can pass the local MIDI review gate. That was useful, but too weak: one lucky sample can hide collapse. This probe records sample-level seeds, grammar pass rates, valid sample rates, and configurable minimum pass thresholds.

This is still a diagnostic tiny-overfit probe. It is not a claim that the model is a musical jazz solo generator.

## Implementation

Code changes:

- `scripts/run_stage_b_generation_probe.py`
  - records `sample_seed` for each generated sample
  - adds `build_probe_summary`
  - reports `valid_sample_rate` and `grammar_gate_sample_rate`
  - supports `--min_valid_samples`
  - supports `--require_all_grammar_samples`
  - supports `--issue_number` for issue-specific reports
- `scripts/agent_harness.sh`
  - adds `stage-b-stronger-probe`
- `tests/test_stage_b_generation_probe.py`
  - verifies multi-sample threshold summaries
  - verifies all-sample grammar gating

## Command

```bash
bash scripts/agent_harness.sh stage-b-stronger-probe
```

Equivalent direct command:

```bash
python scripts/run_stage_b_generation_probe.py \
  --run_id harness_stage_b_stronger_probe \
  --issue_number 24 \
  --max_files 1 \
  --epochs 3 \
  --batch_size 8 \
  --max_sequence 96 \
  --num_samples 3 \
  --generation_mode constrained \
  --constrained_note_groups_per_bar 4 \
  --postprocess_overlap \
  --max_simultaneous_notes 2 \
  --top_k 2 \
  --require_note_groups \
  --require_all_grammar_samples \
  --require_valid_sample \
  --min_valid_samples 1 \
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
outputs/stage_b_generation_probe/harness_stage_b_stronger_probe
```

Dataset/training:

| Metric | Value |
|---|---:|
| role samples | 70 |
| train records | 63 |
| val records | 7 |
| max token id | 544 |
| vocab size | 547 |
| epoch 1 train loss | 6.1389 |
| epoch 1 val loss | 5.5753 |
| epoch 2 train loss | 5.3942 |
| epoch 2 val loss | 5.1037 |
| epoch 3 train loss | 5.1207 |
| epoch 3 val loss | 5.0104 |

Multi-sample gate:

| Metric | Value |
|---|---:|
| samples | 3 |
| grammar gate samples | 3 |
| grammar gate sample rate | 1.000 |
| valid samples | 1 |
| valid sample rate | 0.333 |
| min valid samples | 1 |
| require all grammar samples | true |
| passed grammar gate | true |
| passed generation gate | true |

Sample outcomes:

| Sample | Seed | Valid | Notes After Postprocess | Failure |
|---:|---:|:---:|---:|---|
| 1 | 17 | false | 4 | note count too low: 4 < 6 |
| 2 | 18 | true | 8 | none |
| 3 | 19 | false | 8 | dead-air ratio too high: 0.857 >= 0.800 |

Valid sample #2:

| Metric | Value |
|---|---:|
| note count | 8 |
| unique pitch count | 4 |
| phrase coverage ratio | 0.875 |
| dead-air ratio | 0.714 |
| max simultaneous notes | 2 |
| chord-tone ratio | 0.500 |

## Negative Control

The same 3-epoch checkpoint with `top_k=1` collapsed:

```text
outputs/stage_b_generation_probe/harness_stage_b_stronger_probe_topk1_failure
```

Result:

| Metric | Value |
|---|---:|
| samples | 3 |
| grammar gate samples | 3 |
| valid samples | 0 |
| notes after postprocess | 2 each |
| failure reason | note count too low: 2 < 6 |

This matters because grammar correctness alone is insufficient. The model can emit complete Stage B note groups while still repeating the same onset/pitch enough that postprocess removes most of the phrase.

## Decision

Issue #24 improves the review harness, but the model quality is still weak.

What is now proven:

- Stage B token grammar can be enforced across multiple samples.
- The local probe can report pass rates instead of a single lucky file.
- At least one sample from the stronger setting can pass the current MIDI review gate.

What is not proven:

- stable musical solo-line generation
- good chord following
- useful diversity
- reliable sampling across stricter thresholds

## Next Step

The next issue should stop relying on postprocess alone and improve the generated note distribution:

- add repetition/collapse diagnostics for repeated position-pitch pairs
- add a minimum unique-pitch or unique-position gate before declaring review success
- compare `top_k=1`, `top_k=2`, and a small temperature sweep on the same checkpoint
- consider conditioning or loss changes only after the collapse pattern is measured
