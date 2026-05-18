# Brad Mehldau MIDI Fine-Tuning Probe

This repository is currently focused on symbolic MIDI fine-tuning experiments for jazz piano solo generation.

The active goal is narrow:

> Test whether the current `control_v1` Stage A training path can learn usable solo-line MIDI behavior from the Brad Mehldau MIDI dataset.

This is not currently a Spring Boot/backend MVP, DAW plugin, SaaS product, or realtime performance system. Those ideas are archived until the model path produces reviewable MIDI.

## Current Scope

- Audit the Brad Mehldau MIDI dataset.
- Prepare role-conditioned `control_v1` tokenized data.
- Run tiny and small full-checkpoint training probes.
- Generate MIDI samples from trained checkpoints.
- Reject invalid outputs such as one-note files, long sustain blocks, and chord blocks.
- Decide whether to continue with `control_v1` training or move to duration-explicit tokenization.

## Key Documents

- `docs/CURRENT_STATUS_AND_PLAN.md`
  - Current branch state and next execution order.
- `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
  - Dataset audit, probe order, and acceptance criteria.
- `docs/STAGE_A_TOKEN_FORMAT.md`
  - `control_v1` sequence contract.
- `docs/STAGE_A_TRAINING_MODES.md`
  - Full-checkpoint, adapter, and LoRA mode boundaries.
- `docs/STAGE_A_TINY_OVERFIT.md`
  - Tiny-overfit smoke test criteria.
- `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
  - Review of why previous generated MIDI was not musically valid.
- `docs/REFERENCES.md`
  - 2024-2026 reference map for tokenization, conditioning, sequence length, datasets, and implementation choices.
- `docs/archive/`
  - Deferred backend/API/ERD/realtime/product-planning documents.

## Repository Layout

```text
scripts/
  audit_brad_mehldau_dataset.py   # Dataset audit before training
  prepare_role_dataset.py         # conditioning.mid/target.mid + tokenized data
  control_tokens.py               # control_v1 token helpers
  train_stage_a_full.py           # from-scratch/full-checkpoint training
  train_stage_a_adapter.py        # adapter training when a real base checkpoint exists
  train_qlora.py                  # lower-level training implementation
  generate.py                     # checkpoint-based MIDI generation
  run_control_v1_tiny_overfit.py  # control_v1 tiny-overfit smoke
  agent_harness.sh                # local validation harness

inference/app/
  generator.py                    # request-conditioned generation wrapper
  metrics.py                      # MIDI validity and quality metrics
  postprocess.py                  # repair and output constraints

docs/
  README.md
  CURRENT_STATUS_AND_PLAN.md
  BRAD_MEHLDAU_FINETUNING_PLAN.md
  STAGE_A_*.md
  archive/
```

## Setup

```bash
pip install -r requirements.txt
```

Run the fast local validation harness:

```bash
bash scripts/agent_harness.sh quick
```

## Dataset Audit

```bash
python scripts/audit_brad_mehldau_dataset.py
```

Current local audit result:

- MIDI files: `18`
- usable files: `18`
- files exceeding `max_sequence=512`: `18`
- mean `control_v1_token_count`: about `3931`
- max `control_v1_token_count`: `10653`

Decision:

- Do not train by blindly cropping full songs.
- Use the current control-aware crop path or later build a phrase-window dataset.

## Prepare Data

Small probe:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles_probe2 \
  --role lead \
  --sequence_format control_v1 \
  --max_files 2 \
  --overwrite
```

Full 18-file dataset:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles \
  --role lead \
  --sequence_format control_v1 \
  --overwrite
```

## Train Probe

Start small:

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles_probe2/lead/tokenized \
  --output_dir ./checkpoints/brad_mehldau_control_v1_probe2 \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --max_sequence 512
```

Only after the small probe runs and generates reviewable MIDI, run a larger 5-file or full 18-file training probe.

## Quality Gate

A `.mid` file is not successful just because it exists.

Invalid outputs include:

- one-note or two-note files
- repeated single-pitch phrases
- long sustain blocks
- block chords masquerading as a solo line
- phrase coverage that does not match the requested bars
- excessive dead air for the selected density

If `control_v1` training still produces sustain/chord blocks, the next move is duration-explicit tokenization, not more postprocess.

## Agent Workflow

Project rules live in `AGENTS.md`.

Before committing:

```bash
bash scripts/agent_harness.sh quick
```

For inference, generation, metrics, or model-loading changes, also run:

```bash
bash scripts/agent_harness.sh demo
```

For tiny-overfit or training-mode changes, also run:

```bash
bash scripts/agent_harness.sh tiny-compare
```
