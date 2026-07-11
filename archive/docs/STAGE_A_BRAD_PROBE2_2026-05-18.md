# Stage A Brad Probe 2: 2026-05-18

## Purpose

This probe checks whether the current Stage A `control_v1` path can learn reviewable jazz solo MIDI from two real Brad Mehldau MIDI files.

The goal is not to prove style quality. The goal is narrower:

- verify role dataset preparation from real Brad files
- verify full-checkpoint training runs
- verify generation produces valid MIDI notes
- decide whether to continue `control_v1` broad training or move to Stage B tokenization

## Dataset

Command:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir outputs/issue13_control_v1_brad_probe2/roles_probe2 \
  --role lead \
  --sequence_format control_v1 \
  --max_files 2 \
  --overwrite
```

Result:

| Metric | Value |
|---|---:|
| input files | 2 |
| role samples | 2 |
| train samples | 1 |
| val samples | 1 |
| tokenized train | 1 |
| tokenized val | 1 |

Source files:

| Sample | Source | Conditioning notes | Target notes |
|---|---|---:|---:|
| `000000` | `From This Moment On.midi` | 1122 | 1514 |
| `000001` | `Ron's Place.midi` | 690 | 1007 |

## Training Runs

Both runs used:

- `scripts/train_stage_a_full.py`
- `control_v1`
- batch size `1`
- `max_sequence=192`
- model size `n_layers=2`, `num_heads=4`, `d_model=128`, `dim_feedforward=256`
- seed `17`

### 5 Epoch Probe

Output:

```text
outputs/issue13_control_v1_brad_probe2/checkpoints_probe2_e5
```

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 6.1791 | 6.0357 |
| 5 | 5.5781 | 5.5039 |

### 100 Epoch Probe

Output:

```text
outputs/issue13_control_v1_brad_probe2/checkpoints_probe2_e100
```

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 6.1791 | 6.0357 |
| 10 | 4.8474 | 4.9899 |
| 50 | 2.8408 | 4.4941 |
| 70 | 3.2461 | 4.1306 |
| 100 | 2.5513 | 4.6690 |

Best observed val loss was `4.1306` at epoch `70`.

## Generation Results

The generated files are local artifacts under:

```text
outputs/issue13_control_v1_brad_probe2/
```

They are not committed.

### 5 Epoch, `top_k=1`

All three generated samples had zero notes.

| File | Valid | Reason | Notes |
|---|---|---|---:|
| `jazz_sample_1.mid` | no | generated MIDI has no notes | 0 |
| `jazz_sample_2.mid` | no | generated MIDI has no notes | 0 |
| `jazz_sample_3.mid` | no | generated MIDI has no notes | 0 |

### 5 Epoch, `top_k=32`

The samples were still not reviewable.

| File | Valid | Reason | Notes |
|---|---|---|---:|
| `jazz_sample_1.mid` | no | generated MIDI has no notes | 0 |
| `jazz_sample_2.mid` | no | note count too low | 1 |
| `jazz_sample_3.mid` | no | note count too low | 2 |

### 100 Epoch, `top_k=1`

All three deterministic samples collapsed to a one-note phrase.

| File | Valid | Reason | Notes | Unique pitches |
|---|---|---|---:|---:|
| `jazz_sample_1.mid` | no | note count too low | 1 | 1 |
| `jazz_sample_2.mid` | no | note count too low | 1 | 1 |
| `jazz_sample_3.mid` | no | note count too low | 1 | 1 |

### 100 Epoch, `top_k=32`

This was the strongest run, but it still failed the Stage A review gate.

| File | Valid | Reason | Notes | Unique pitches | Coverage | Max duration ratio | Max simultaneous |
|---|---|---|---:|---:|---:|---:|---:|
| `jazz_sample_1.mid` | no | note duration too long | 6 | 6 | 1.000 | 0.705 | 3 |
| `jazz_sample_2.mid` | no | note duration too long | 6 | 5 | 0.832 | 0.677 | 3 |
| `jazz_sample_3.mid` | no | note count too low | 5 | 5 | 1.000 | 0.276 | 2 |

`jazz_sample_3.mid` previously slipped through the medium numeric gate because the old minimum was only four notes for two bars. The gate now requires at least six notes for a two-bar medium phrase.

## Quality Gate Change

Changed medium density minimum note count from `2.0` notes per bar to `3.0` notes per bar.

For the default two-bar phrase:

- old medium minimum: 4 notes
- new medium minimum: 6 notes

This prevents five-note two-bar MIDI files from being treated as reviewable solo-line samples.

## Decision

The current Stage A `control_v1` path is validated as a runnable pipeline, but not as a usable jazz solo generator.

What worked:

- real Brad files can be converted into role-conditioned samples
- tokenized train/val records are produced
- full-checkpoint training runs locally
- checkpoint generation runs
- metrics and gates catch empty, one-note, long-duration, and too-sparse outputs

What failed:

- 5-epoch generation produced no usable notes
- 100-epoch deterministic generation collapsed to one note
- sampled generation still produced long sustain blocks or too few notes
- the output is not a solo line and is not useful for musical review

## Next Move

Do not expand to broad generic jazz training on the current `control_v1` representation yet.

The next technical issue should be Stage B tokenization:

- explicit `BAR`
- explicit `POSITION`
- explicit `CHORD`
- explicit `NOTE_PITCH`
- explicit `NOTE_DURATION`
- optional `VELOCITY`
- phrase/window dataset instead of full-song target continuation

The failure mode points to representation and dataset-windowing, not to another postprocess tweak.
