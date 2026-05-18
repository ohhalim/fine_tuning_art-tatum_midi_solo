# Stage A Tiny Overfit Smoke

## Goal

Verify whether the current Stage A model stack can learn basic symbolic MIDI solo grammar before adding broader conditioning, realtime runtime, or backend scope.

This test is intentionally small:

- 1-3 deterministic MIDI solo phrases
- existing NOTE_ON / NOTE_OFF / TIME_SHIFT / VELOCITY token stream
- small Music Transformer architecture
- full checkpoint loading through the same generation path
- MVP metrics gate to catch one-note, long-sustain, and chord-block failures

## Command

Recommended smoke:

```bash
python scripts/run_stage_a_tiny_overfit.py \
  --sample_count 3 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 128 \
  --primer_max_tokens 24
```

Fast wiring check:

```bash
python scripts/run_stage_a_tiny_overfit.py \
  --sample_count 1 \
  --epochs 1 \
  --max_sequence 96 \
  --primer_max_tokens 16
```

Compare full-model tiny training and random-base LoRA-only under identical settings:

```bash
python scripts/compare_stage_a_tiny_modes.py \
  --sample_count 3 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 128 \
  --primer_max_tokens 24
```

## Outputs

Each run writes to:

```text
outputs/stage_a_tiny_overfit/<run_id>/
  input_midi/
  tokenized/
    train/
    val/
  checkpoints/
  raw_samples/
  generated/
  tiny_dataset_manifest.json
  report.json
  report.md
```

Comparison runs write:

```text
outputs/stage_a_tiny_compare/<run_id>/
  comparison.json
  comparison.md
```

## Decision Rule

Continue the current Stage A tokenization/training path only if at least one fixed-seed tiny-overfit generation passes the MVP gate with:

- `status=COMPLETED`
- `fallback_used=false`
- non-trivial note count
- acceptable max note duration ratio
- no one-note/two-note output

If the tiny-overfit run still falls back, do not expand conditioning yet. The next step should be to inspect whether the blocker is:

- LoRA-only random-base training
- NOTE_ON/OFF grammar instability
- autoregressive generation sampling
- decoder/postprocess assumptions

The script exits with code `2` when the smoke completes but the model path does not pass the MVP gate. In that case, inspect `report.md` and `report.json`; the run itself is still useful diagnostic output.

## Implementation Notes

`scripts/train_qlora.py --train_full_model` keeps the LoRA-wrapped checkpoint format but unfreezes the base model. This is deliberate: tiny overfit is a diagnostic, not the final training recipe. It separates "the stack can learn MIDI grammar" from "LoRA-only adapter training from a random base is enough."

New checkpoints include `model_config`, so `scripts/generate.py` can reconstruct small overfit architectures without manually passing `n_layers`, `d_model`, or related flags.

## Local Result: 2026-05-18

Command:

```bash
python scripts/run_stage_a_tiny_overfit.py \
  --sample_count 3 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 128 \
  --primer_max_tokens 24 \
  --num_samples 3 \
  --output_root outputs/stage_a_tiny_overfit_test \
  --run_id dense_overfit_200
```

Result:

- training reached `best validation loss: 0.0568`
- raw model samples passed the medium MIDI gate
- MVP inference gate returned `status=COMPLETED`
- `fallback_used=false`
- repaired inference output metrics:
  - `note_count=15`
  - `note_density=4.61`
  - `dead_air_ratio=0.36`
  - `max_note_duration_ratio=0.496`
  - `max_simultaneous_notes=2`
  - `chord_tone_ratio=0.73`

Decision:

The stack can learn basic MIDI grammar when the tiny overfit uses full-model training and a dense enough known-good phrase. Continue with the current event token path for one more controlled experiment, but do not treat LoRA-only random-base training as validated.

Comparison:

- `--lora_only` with the same 3 samples / 200 epochs reached only `best validation loss: 4.8228`.
- All raw model samples decoded to `note_count=0`.
- MVP inference fell back with `model_failure_reason=generated MIDI has no notes`.

Conclusion:

The original random-base LoRA-only Stage A training path is not sufficient even for this tiny MIDI grammar smoke. The next model work should either train the full symbolic model, start from a real pretrained base, or explicitly separate adapter training from from-scratch training.
