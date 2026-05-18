# Jazz Piano MIDI Fine-Tuning Probe

This repository is currently focused on symbolic MIDI fine-tuning experiments for jazz piano solo generation.

The active goal is narrow:

> Audit the full jazz piano MIDI corpus, validate the current `control_v1` Stage A path, and decide whether to build a generic jazz pianist base before Brad Mehldau style adaptation.

This is not currently a Spring Boot/backend MVP, DAW plugin, SaaS product, or realtime performance system. Those ideas are archived until the model path produces reviewable MIDI.

## Current Scope

- Audit the full jazz piano MIDI dataset.
- Keep Brad Mehldau data separate for style adaptation and holdout evaluation.
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
- `docs/DATASET_STRATEGY.md`
  - Full jazz piano corpus audit, generic jazz pianist base, and Brad style adaptation plan.
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
  audit_jazz_piano_dataset.py     # Full corpus audit before broad training
  build_jazz_training_manifests.py # Audit-based train/val/holdout manifest builder
  audit_brad_mehldau_dataset.py   # Dataset audit before training
  prepare_role_dataset.py         # conditioning.mid/target.mid + tokenized data
  control_tokens.py               # control_v1 token helpers
  train_stage_a_full.py           # from-scratch/full-checkpoint training
  train_stage_a_adapter.py        # adapter training when a real base checkpoint exists
  train_qlora.py                  # lower-level training implementation
  generate.py                     # checkpoint-based MIDI generation
  run_control_v1_tiny_overfit.py  # control_v1 tiny-overfit smoke
  run_manifest_prepare_smoke.py   # audit -> manifest -> prepare dry-run
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
python scripts/audit_jazz_piano_dataset.py
```

Current local audit result:

- active dataset tree: `midi_dataset/midi`
- files: `2777`
- readable files: `2777`
- candidate files: `2775`
- candidate non-Brad files: `2703`
- candidate Brad files: `72`
- exact duplicate hash groups: `0`

Decision:

- Use the filtered non-Brad candidates for a generic jazz pianist base only after tokenizer sanity is validated.
- Use Brad candidates for style adaptation and holdout evaluation.
- Do not train both `midi_dataset/midi` and duplicate mirror `midi_dataset/midi_kong`.

Build training manifests from the audit JSON:

```bash
python scripts/build_jazz_training_manifests.py
```

Generated manifest files are written under `data/manifests/` and are not committed.

Run a small end-to-end dry-run:

```bash
bash scripts/agent_harness.sh manifest-dry-run
```

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

Manifest-based generic split:

```bash
python scripts/prepare_role_dataset.py \
  --train_manifest ./data/manifests/generic_jazz_train.txt \
  --val_manifest ./data/manifests/generic_jazz_val.txt \
  --output_dir ./data/roles_generic_jazz \
  --role lead \
  --sequence_format control_v1 \
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
