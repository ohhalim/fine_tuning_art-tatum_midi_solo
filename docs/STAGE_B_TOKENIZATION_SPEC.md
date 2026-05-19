# Stage B Tokenization Spec

작성일: 2026-05-19

## Purpose

Stage B replaces the Stage A `NOTE_ON` / `NOTE_OFF` continuation format for the next model probe.

The Stage A Brad 2-file probe proved that the pipeline can run, but generation still collapsed into empty MIDI, one-note phrases, sparse fragments, or long sustain blocks. The next fix is not more postprocess. The next fix is an explicit symbolic representation.

Stage B must make these musical facts directly visible to the model:

- bar boundary
- position inside bar
- chord context
- note pitch
- note duration
- velocity

## Non-Goals

Do not implement these in the first Stage B issue:

- broad generic jazz training
- full 18-file Brad training
- DAW/realtime integration
- natural-language prompting
- artist-clone product wording
- compound-word Transformer architecture

The first objective is a tokenizer contract and tiny roundtrip tests.

## Format Name

```text
stage_b_v1
```

## Token Families

Stage B keeps existing Stage A control tokens for role and tempo, then adds a separate token range after the Stage A control-token range.

Current implementation file:

```text
scripts/stage_b_tokens.py
```

Core families:

| Family | Meaning |
|---|---|
| `BAR` | Reuses existing `TOKEN_BAR` |
| `POSITION_*` | 16th-note grid position inside a 4/4 bar |
| `CHORD_ROOT_*` | `C`, `C#`, `D`, ..., `B`, or `N` |
| `CHORD_QUALITY_*` | `maj`, `min`, `dom7`, `maj7`, `min7`, `dim`, `halfdim`, `sus`, `unknown` |
| `NOTE_PITCH_*` | Piano pitch range `21..108` |
| `NOTE_DURATION_*` | Explicit duration in 16th-note steps |
| `VELOCITY_*` | 8 velocity bins |
| `END` | Existing `TOKEN_END` |

## Sequence Layout

For a target phrase:

```text
ROLE_LEAD
TEMPO_*
BAR
CHORD_ROOT_*
CHORD_QUALITY_*
POSITION_*
VELOCITY_*
NOTE_PITCH_*
NOTE_DURATION_*
...
BAR
CHORD_ROOT_*
CHORD_QUALITY_*
...
END
```

Notes are encoded as a four-token group:

```text
POSITION_* VELOCITY_* NOTE_PITCH_* NOTE_DURATION_*
```

This is intentionally simple. It is not yet optimized for sequence length. The immediate goal is to remove ambiguous `NOTE_OFF` learning from the model path.

## Quantization

Initial defaults:

| Setting | Value |
|---|---:|
| time signature | `4/4` only |
| positions per bar | `16` |
| max duration steps | `16` |
| velocity bins | `8` |
| pitch range | `A0..C8` / MIDI `21..108` |

At 120 BPM:

- 1 bar = 2.0 seconds
- 1 position step = 0.125 seconds
- a 0.25 second note becomes `NOTE_DURATION_2`

Long durations are clamped at `NOTE_DURATION_16` in the tokenizer. Musical validity is still checked later by metrics gates.

## Chord Handling

Chord symbols are parsed into root and quality tokens.

Examples:

| Chord | Tokens |
|---|---|
| `Cm7` | `CHORD_ROOT_C`, `CHORD_QUALITY_min7` |
| `F7` | `CHORD_ROOT_F`, `CHORD_QUALITY_dom7` |
| `Bbmaj7` | `CHORD_ROOT_Bb`, `CHORD_QUALITY_maj7` |
| `F#m7b5` | `CHORD_ROOT_F#`, `CHORD_QUALITY_halfdim` |
| missing/unknown | `CHORD_ROOT_N`, `CHORD_QUALITY_unknown` |

For request-conditioned generation, chords come from the request.

For raw MIDI dataset preparation, chord labels may be unavailable. The first Stage B dataset builder can use `N/unknown` for dataset-only probes, then add chord inference or lead-sheet metadata later.

## Why This Should Fix The Current Failure

Stage A failure:

- note duration depends on correct future `NOTE_OFF`
- bar position is implicit in accumulated `TIME_SHIFT`
- chord context is mostly hidden in primer MIDI
- long full-song streams force cropping and weak local context

Stage B response:

- duration is a direct token
- onset position is a direct token
- chord is a direct token
- phrase/window data can be short and stable
- decoder can build notes without pairing `NOTE_ON` and `NOTE_OFF`

## Acceptance Criteria

Issue #14 is complete when:

- `scripts/stage_b_tokens.py` defines stable Stage B token ranges.
- Stage B token IDs do not overlap Stage A control tokens.
- A small note list can encode into `stage_b_v1`.
- The encoded sequence includes `BAR`, `POSITION`, `CHORD`, `NOTE_PITCH`, `NOTE_DURATION`, and `VELOCITY`.
- A simple roundtrip test decodes quantized notes with correct pitch/start/end.
- Current Stage A tests still pass.

## Dataset Preparation

Issue #15 wires `stage_b_v1` into the existing role dataset preparation entrypoint.

Example:

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir outputs/issue15_stage_b_probe2/roles_stage_b_probe2 \
  --role lead \
  --sequence_format stage_b_v1 \
  --max_files 2 \
  --overwrite
```

The first dataset contract is target-only:

- `conditioning.mid` is still written for compatibility with the role dataset folder shape.
- tokenized `.npy` records contain `stage_b_v1` target-note tokens.
- records do not include `COND_SEP`.
- chord labels are `N/unknown` unless explicit chord metadata is added later.

Phrase/window extraction:

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

This produces short tokenized phrase windows rather than full-song target continuations.

## Next Issue After Spec

After the phrase/window dataset contract is merged, the next issue should build:

- target-only training loss if conditioning is prepended
- Stage B tiny-overfit smoke
- Brad 2-file Stage B probe
