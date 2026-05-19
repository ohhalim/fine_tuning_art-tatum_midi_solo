# Stage B Window Tiny-Overfit Smoke: 2026-05-19

## Purpose

Issue #17 connects the Stage B phrase/window dataset path to the model training path.

Issue #16 proved that Brad MIDI files can be split into short 2-bar Stage B token records. The next risk was model compatibility: Stage B token IDs are higher than the old Stage A/control-token vocabulary, so training would fail or learn against the wrong output size unless the model vocabulary is expanded.

This issue is still not a musical success claim. It only proves that short Stage B records can be prepared, checked against model `VOCAB_SIZE`, and passed into a tiny full-model training smoke.

## Implementation

Code changes:

- `music_transformer/utilities/constants.py`
  - defines shared Stage B token ranges
  - sets `VOCAB_SIZE = TOKEN_STAGE_B_END + 1`
- `scripts/stage_b_tokens.py`
  - reuses the shared Stage B constants instead of maintaining independent ranges
  - keeps existing aliases such as `TOKEN_POSITION_START` for tokenizer code
- `scripts/run_stage_b_window_tiny_overfit.py`
  - prepares Stage B phrase windows through `prepare_role_dataset.py`
  - calculates token length and max token id stats
  - fails if no tokenized records are produced
  - fails if any Stage B token id exceeds model `VOCAB_SIZE`
  - optionally runs the existing full-model tiny training command
- `scripts/agent_harness.sh`
  - adds `stage-b-window-prepare` mode for a fast local Stage B prepare/vocab check
- `tests/test_stage_b_tokens.py`
  - asserts `STAGE_B_VOCAB_SIZE == VOCAB_SIZE`
- `tests/test_stage_b_window_tiny_overfit.py`
  - verifies empty tokenized datasets are not treated as successful
  - verifies max-token-id checks accept tokens inside model vocab

## Commands

Prepare-only smoke:

```bash
python scripts/run_stage_b_window_tiny_overfit.py \
  --run_id local_prepare_smoke \
  --max_files 1 \
  --prepare_only
```

One-epoch training smoke:

```bash
python scripts/run_stage_b_window_tiny_overfit.py \
  --run_id local_train_smoke_e1 \
  --max_files 1 \
  --epochs 1 \
  --batch_size 8 \
  --max_sequence 128 \
  --n_layers 1 \
  --num_heads 4 \
  --d_model 64 \
  --dim_feedforward 128 \
  --lora_r 4 \
  --lora_alpha 8
```

Harness check:

```bash
bash scripts/agent_harness.sh stage-b-window-prepare
```

## Local Result

Prepare-only smoke on one Brad Mehldau studio MIDI file:

| Metric | Value |
|---|---:|
| role samples | 70 |
| token files | 70 |
| min token length | 33 |
| p50 token length | 89 |
| max token length | 212 |
| mean token length | 95.74 |
| max token id | 544 |
| model vocab size | 547 |
| fits vocab | true |

One-epoch tiny training smoke:

| Metric | Value |
|---|---:|
| train records | 63 |
| val records | 7 |
| epoch | 1 |
| train loss | 6.1135 |
| val loss | 5.8195 |
| return code | 0 |

Checkpoint output:

```text
outputs/stage_b_window_tiny_overfit/harness_stage_b_window_train_e1/checkpoints
```

Generated outputs remain local artifacts and are not committed.

## Decision

Stage B window records now fit the model vocabulary, and the training entrypoint accepts them.

This only validates the data/model plumbing. It does not prove that the model can generate reviewable jazz MIDI yet.

Next issue should add a Stage B decode/generation probe:

- train a tiny Stage B checkpoint for enough epochs to overfit a tiny window set
- decode generated Stage B tokens back to MIDI
- reject one-note, two-note, long-sustain, and chord-block outputs
- compare generated MIDI against the same review gates used after the Stage A failure
