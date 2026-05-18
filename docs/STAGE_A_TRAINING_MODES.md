# Stage A Training Modes

## Purpose

Stage A now separates from-scratch full-model training from adapter training.

This separation exists because the tiny-overfit comparison showed:

- full-model tiny training can learn basic MIDI grammar
- random-base LoRA-only training does not learn even the tiny smoke reliably

Therefore, do not use random-base LoRA-only training as the default Stage A path.

## Mode 1: Full Checkpoint / From-Scratch Training

Use this when no real pretrained symbolic MIDI base exists.

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles/lead/tokenized \
  --output_dir ./checkpoints/stage_a_full_scratch \
  --epochs 3 \
  --batch_size 8 \
  --max_sequence 512
```

This wrapper calls:

```bash
python scripts/train_qlora.py ... --train_full_model
```

Important behavior:

- LoRA modules are still attached so checkpoint shape stays compatible with the current generation loader.
- Base transformer, embeddings, output head, and LoRA modules are all trainable.
- `checkpoint_epoch*.pt` is the primary artifact.
- `lora_weights.pt` is not enough to reconstruct a trained model by itself.

## Mode 2: Adapter Training From Pretrained/Base Checkpoint

Use this only when a meaningful base checkpoint exists.

```bash
python scripts/train_stage_a_adapter.py \
  --checkpoint ./checkpoints/stage_a_full_scratch/checkpoint_epoch3.pt \
  --data_dir ./data/roles/lead/tokenized \
  --output_dir ./checkpoints/stage_a_adapter \
  --epochs 3 \
  --batch_size 8 \
  --max_sequence 512
```

Important behavior:

- `--checkpoint` is required.
- The base model remains frozen after LoRA modules are attached.
- Full training checkpoints with `model_state_dict` are supported.
- Base-only state dict checkpoints are supported.
- LoRA-only `lora_weights.pt` is rejected as an adapter base.

## Legacy Entry Point

`scripts/train_qlora.py` remains the lower-level implementation used by both wrappers.

Direct random-base LoRA-only calls are allowed for diagnostics, but should not be treated as a valid Stage A training strategy:

```bash
python scripts/train_qlora.py --data_dir ./data/roles/lead/tokenized
```

If this path is used, report it honestly as `random_base_lora`.

## Current Decision

For the next Stage A model run:

1. Use `scripts/train_stage_a_full.py` unless a real pretrained symbolic MIDI base is available.
2. Use `scripts/train_stage_a_adapter.py` only with an explicit full/base checkpoint.
3. Continue measuring generated MIDI through the MVP gate; do not trust loss alone.
