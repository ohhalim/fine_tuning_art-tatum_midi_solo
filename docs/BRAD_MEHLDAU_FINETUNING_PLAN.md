# Brad Mehldau Fine-Tuning Plan

## Goal

Fine-tune the Stage A symbolic MIDI model on the Brad Mehldau MIDI dataset.

This branch is not about backend/API/realtime work. The goal is to determine whether the current `control_v1` symbolic training path can learn usable jazz piano solo MIDI from the Brad Mehldau dataset.

## Dataset Audit

Command:

```bash
python scripts/audit_brad_mehldau_dataset.py
```

Local audit result:

| Metric | Value |
|---|---:|
| MIDI files | 18 |
| usable files | 18 |
| unusable files | 0 |
| max_sequence | 512 |
| files exceeding max_sequence | 18 |

Token stats:

| Metric | Min | P50 | P90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| `control_v1_token_count` | 1136 | 3241 | 5663 | 10653 | 3931.39 |
| `conditioning_token_count` | 468 | 1608 | 2843 | 4550 | 1937.22 |
| `target_token_count` | 419 | 1716 | 2894 | 6098 | 1989.17 |
| `note_count` | 266 | 756 | 1286 | 2636 | 942.33 |

Result:

- All Brad Mehldau MIDI files are usable under the current split thresholds.
- Every file exceeds `max_sequence=512`.
- Full-song sequence training cannot rely on plain random crop.
- The current control-aware crop fix is required so `ROLE_LEAD + TEMPO_* + BAR + conditioning_tail + COND_SEP` stays visible during training.

## Current Training Path

Use:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles \
  --role lead \
  --sequence_format control_v1 \
  --overwrite
```

Then:

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles/lead/tokenized \
  --output_dir ./checkpoints/brad_mehldau_control_v1_stage_a \
  --epochs 3 \
  --batch_size 8 \
  --num_workers 4 \
  --max_sequence 512
```

## Probe Order

Do not jump straight to a long full training run.

1. `max_files=2` prepare probe.
2. `max_files=2` short full-model training probe.
3. `max_files=5` training probe.
4. Full 18-file training run.
5. Generate raw samples and run MVP gate.
6. Inspect piano roll/listen before claiming improvement.

## Acceptance Criteria

A training probe is useful only if:

- training runs without loader/checkpoint errors
- generated MIDI is not empty
- fallback is not used for accepted model output
- note count and phrase coverage pass gate
- max note duration ratio passes gate
- max simultaneous notes passes solo-line gate
- output is not a long sustain block or chord block

If the model still generates sustain/chord blocks after `control_v1` training, move to duration-explicit tokenization instead of adding more postprocess.
