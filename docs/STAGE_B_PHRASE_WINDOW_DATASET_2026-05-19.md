# Stage B Phrase Window Dataset: 2026-05-19

## Purpose

Build short Stage B phrase/window records instead of training on full target continuations.

Issue #15 made `stage_b_v1` usable from `prepare_role_dataset.py`, but the first Brad dry run still produced full target sequences with thousands of tokens. Issue #16 adds fixed-bar windows so the next training probe can learn short musical phrases first.

## Implementation

Updated:

- `scripts/prepare_role_dataset.py`
- `tests/test_stage_b_tokens.py`

New options:

```text
--stage_b_window_bars
--stage_b_window_stride_bars
--stage_b_min_window_target_notes
```

These options are only meaningful with:

```text
--sequence_format stage_b_v1
```

Behavior:

- target notes are sliced into fixed bar windows
- note times are normalized to the window start
- sparse conditioning MIDI is still written for folder compatibility
- tokenized records remain target-only Stage B sequences
- manifest train/val boundaries are preserved

## Local Brad 2-File Window Dry Run

Command:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir outputs/issue16_stage_b_window_probe2/roles_stage_b_window_probe2 \
  --role lead \
  --sequence_format stage_b_v1 \
  --stage_b_window_bars 2 \
  --stage_b_window_stride_bars 2 \
  --stage_b_min_window_target_notes 4 \
  --max_files 2 \
  --overwrite
```

Result:

| Metric | Value |
|---|---:|
| input files | 2 |
| role samples | 137 |
| train samples | 123 |
| val samples | 14 |
| tokenized train | 123 |
| tokenized val | 14 |

Token length stats:

| Metric | Tokens |
|---|---:|
| min | 22 |
| p50 | 77 |
| max | 212 |
| mean | 82.94 |

Compared with the previous full-target Stage B dry run:

| Run | Train tokens | Val tokens |
|---|---:|---:|
| full target | 4430 | 6482 |
| 2-bar windows | p50 77 | p50 77 |

Example token head:

```text
ROLE_LEAD
TEMPO_DANCE
BAR
CHORD_ROOT_N
CHORD_QUALITY_unknown
POSITION_9
VELOCITY_4
NOTE_PITCH_63
NOTE_DURATION_4
```

Generated output artifacts remain local under:

```text
outputs/issue16_stage_b_window_probe2/
```

## Validation

```bash
python -m unittest tests.test_stage_b_tokens
bash scripts/agent_harness.sh quick
```

## Decision

Stage B now has a short phrase/window dataset path.

The next issue can run a Stage B tiny-overfit using these short tokenized examples, but should still avoid broad generic jazz training until tiny-overfit generation decodes to reviewable MIDI.
