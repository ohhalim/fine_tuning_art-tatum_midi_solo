# Stage B Collapse Diagnostics and Sampling Sweep: 2026-05-20

## Purpose

Issue #29 implements `CORE_PLAN.md` Phase 1-2.

Issue #24 showed that Stage B constrained generation can pass grammar and occasionally pass the full MIDI review gate, but it also exposed a collapse mode:

- `top_k=1` generated complete note groups
- the groups repeated the same position/pitch
- overlap postprocess removed most notes
- the sample failed as a solo-line candidate

This issue makes that failure measurable instead of only visible in a piano roll.

## Implementation

Code changes:

- `scripts/run_stage_b_generation_probe.py`
  - adds token-level note group extraction
  - adds collapse diagnostics per sample
  - records repeated pitch ratio
  - records repeated position/pitch pair ratio
  - records postprocess removal ratio
  - records diagnostic failure reasons that include collapse causes
- `scripts/run_stage_b_sampling_sweep.py`
  - trains one tiny Stage B checkpoint
  - reuses the checkpoint across sampling configs
  - writes `sweep_report.json`
  - writes `sweep_report.md`
- `scripts/agent_harness.sh`
  - adds `stage-b-collapse-sweep`
- tests
  - verify collapse detection
  - verify summary aggregation
  - verify sweep summary selection

## Commands

```bash
bash scripts/agent_harness.sh stage-b-collapse-sweep
```

Equivalent sweep:

```bash
python scripts/run_stage_b_sampling_sweep.py \
  --run_id harness_stage_b_collapse_sweep \
  --issue_number 29 \
  --top_ks 1,2 \
  --temperatures 0.9 \
  --train_top_k 2 \
  --max_files 1 \
  --epochs 3 \
  --batch_size 8 \
  --max_sequence 96 \
  --num_samples 3 \
  --constrained_note_groups_per_bar 4 \
  --max_simultaneous_notes 2 \
  --require_all_grammar_samples \
  --min_best_valid_samples 1 \
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
outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep
```

Sweep summary:

| top_k | temp | samples | grammar | valid | valid rate | collapse rate | avg pair repeat | max removal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.900 | 3 | 3 | 0 | 0.000 | 1.000 | 0.875 | 0.750 |
| 2 | 0.900 | 3 | 3 | 1 | 0.333 | 0.333 | 0.292 | 0.500 |

Best config:

```text
top_k=2, temperature=0.9
```

Diagnostic failures:

- `top_k=1`
  - `note count too low: 2 < 6`
  - collapse: `single_pitch`, `single_position`, `repeated_position_pitch`, `postprocess_removed_majority`
- `top_k=2`
  - one sample failed with note count too low plus repeated position/pitch collapse
  - one sample failed on dead-air ratio
  - one sample passed the current review gate

## Decision

Issue #29 confirms that the Stage B grammar path is not the bottleneck anymore.

The bottleneck is note distribution:

- deterministic decoding collapses to a single position/pitch pattern
- mild stochasticity improves the result but is not stable
- postprocess hides some overlap, but cannot create a real solo-line

Do not proceed to broad generic jazz training yet.

## Next Step

The next issue should tighten the review gate before scaling data:

- add minimum unique-position or unique-position/pitch thresholds to the review gate
- require collapse warning rate below a configured threshold
- keep sampling sweep as the comparison tool

After that, run a Stage B 2-file Brad probe only if the stricter gate is not immediately failing every sample.
