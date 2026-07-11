# Stage B Role Dataset Prep: 2026-05-19

## Purpose

Wire `stage_b_v1` into the existing role dataset preparation path.

This is not a training run. The goal is to prove that real MIDI files can become Stage B tokenized train/val records before building phrase/window extraction.

## Implementation

Updated:

- `scripts/prepare_role_dataset.py`
- `scripts/stage_b_tokens.py`
- `tests/test_stage_b_tokens.py`

New prepare mode:

```bash
python scripts/prepare_role_dataset.py \
  --sequence_format stage_b_v1
```

The first Stage B dataset contract is target-only:

- `conditioning.mid` is still written for folder compatibility.
- tokenized `.npy` records encode `target.mid`.
- records use explicit `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups.
- records do not include `COND_SEP`.
- chord labels are `N/unknown` until chord metadata or inference is added.

## Local Brad 2-File Dry Run

Command:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir outputs/issue15_stage_b_probe2/roles_stage_b_probe2 \
  --role lead \
  --sequence_format stage_b_v1 \
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

Token lengths:

| Split | File | Tokens |
|---|---|---:|
| train | `000000.npy` | 4430 |
| val | `000000.npy` | 6482 |

Example token head:

```text
ROLE_LEAD
TEMPO_DANCE
BAR
CHORD_ROOT_N
CHORD_QUALITY_unknown
POSITION_11
VELOCITY_3
NOTE_PITCH_63
NOTE_DURATION_2
```

Generated output artifacts remain local under:

```text
outputs/issue15_stage_b_probe2/
```

## Validation

```bash
python -m unittest tests.test_stage_b_tokens tests.test_control_tokens
bash scripts/agent_harness.sh quick
```

Both passed.

## Decision

`stage_b_v1` can now produce train/val `.npy` files from the existing role dataset builder.

Next work should not train the model yet. The next issue should build phrase/window extraction so Stage B examples are short musical phrases instead of full target continuations.
