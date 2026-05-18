# Stage A Token Format

## Purpose

Stage A uses symbolic MIDI event tokens plus a small control-token prefix.

The goal is to stop using `TOKEN_END` as both "conditioning separator" and "end of sequence". `TOKEN_END` now means sequence end only in the current `control_v1` format.

## control_v1

Training sequence:

```text
ROLE_LEAD + TEMPO_* + BAR + conditioning_tokens + COND_SEP + target_tokens + END
```

Generation primer:

```text
ROLE_LEAD + TEMPO_* + BAR + conditioning_tokens + COND_SEP
```

Current control tokens:

| Token | Meaning |
|---|---|
| `ROLE_LEAD` | lead/solo target role |
| `TEMPO_SLOW` | bpm < 90 |
| `TEMPO_MEDIUM` | 90 <= bpm < 120 |
| `TEMPO_DANCE` | 120 <= bpm < 150 |
| `TEMPO_FAST` | bpm >= 150 |
| `BAR` | coarse bar-level prompt marker |
| `COND_SEP` | conditioning/target boundary |

## Legacy Format

Legacy Stage A experiments used:

```text
conditioning_tokens + END + target_tokens + END
```

Use this only when reproducing older checkpoints:

```bash
python scripts/prepare_role_dataset.py --sequence_format legacy_sep
python scripts/generate.py --control_format legacy_sep
```

## Checkpoint Compatibility

Control tokens expand the model vocabulary. New loaders resize old checkpoint token layers when possible:

- `embedding.weight`
- `Wout.weight`
- `Wout.bias`

Old rows are copied into the new tensors and new control-token rows keep the model's fresh initialization. This keeps legacy checkpoints loadable, but they have not learned the new control tokens.

## Long Sequence Cropping

Real role-conditioned MIDI files can be thousands of tokens long. A Brad Mehldau `max_files=2` prepare probe produced a first train sequence of `7079` tokens.

For `control_v1`, training crop must not use plain random windows because most windows would drop the control prompt. `scripts/train_qlora.py` now preserves:

```text
ROLE_LEAD + TEMPO_* + BAR + conditioning_tail + COND_SEP + target_window
```

The default conditioning tail budget is `64` tokens. This keeps the prompt contract visible during training while still allowing random target windows.

## Tiny Smoke

Use this before larger Stage A training:

```bash
python scripts/run_control_v1_tiny_overfit.py \
  --sample_count 1 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 192 \
  --primer_max_tokens 96
```

Current local smoke result:

- run id: `control_v1_auto`
- best validation loss: `0.0142`
- raw valid samples: `3/3`
- MVP inference: `COMPLETED`
- fallback used: `false`
