# Stage B Generation Probe: 2026-05-19

## Purpose

Issue #18 adds the first Stage B decode/generation probe.

Issue #17 proved that Stage B phrase/window records fit the model vocabulary and can enter the training path. That was still only data/model plumbing. Issue #18 checks the next boundary:

- can the model generation path emit Stage B token IDs?
- can generated Stage B tokens decode back to MIDI?
- does the decoded MIDI pass the same review gate that rejected Stage A outputs?

This is not a musical success claim. The first smoke is expected to be weak because it trains only one epoch on one Brad file subset.

## Important Finding

Before this issue, `MusicTransformer.generate()` sampled only logits below `TOKEN_END`.

That was acceptable for legacy Stage A event tokens, but Stage B tokens live above the Stage A control-token range. In practice, the model could train on Stage B token IDs but generation could not emit them.

Issue #18 adds:

```text
sample_vocab_size
```

Default behavior still samples up to `TOKEN_END` for older Stage A generation. Stage B probe passes `sample_vocab_size=VOCAB_SIZE`, so the model can sample the full Stage B vocabulary.

## Implementation

Code changes:

- `music_transformer/model/music_transformer.py`
  - adds optional `sample_vocab_size`
  - keeps default `TOKEN_END` behavior for Stage A compatibility
  - lets Stage B generation sample up to `VOCAB_SIZE`
- `scripts/generate.py`
  - forwards optional `sample_vocab_size` to model generation
- `scripts/run_stage_b_generation_probe.py`
  - prepares Stage B phrase windows
  - optionally trains a tiny full-model checkpoint
  - builds a Stage B primer from role, tempo, first bar, and chord tokens
  - samples Stage B tokens with `sample_vocab_size=VOCAB_SIZE`
  - decodes tokens through `decode_stage_b_midi`
  - computes existing MIDI metrics and gate result
  - writes `report.json`
- `scripts/agent_harness.sh`
  - adds `stage-b-generation-probe`
- `tests/test_stage_b_generation_probe.py`
  - verifies primer shape
  - verifies generation uses full `VOCAB_SIZE`
  - verifies Stage B tokens write a MIDI note

## Command

```bash
bash scripts/agent_harness.sh stage-b-generation-probe
```

Equivalent direct command:

```bash
python scripts/run_stage_b_generation_probe.py \
  --run_id harness_stage_b_generation_probe \
  --max_files 1 \
  --epochs 1 \
  --batch_size 8 \
  --max_sequence 96 \
  --num_samples 1 \
  --n_layers 1 \
  --num_heads 4 \
  --d_model 64 \
  --dim_feedforward 128 \
  --lora_r 4 \
  --lora_alpha 8
```

## Local Result

Output:

```text
outputs/stage_b_generation_probe/harness_stage_b_generation_probe
```

Dataset/training:

| Metric | Value |
|---|---:|
| role samples | 70 |
| train records | 63 |
| val records | 7 |
| max token id | 544 |
| sample vocab size | 547 |
| epoch 1 train loss | 6.2115 |
| epoch 1 val loss | 5.9441 |

Generation:

| Metric | Value |
|---|---:|
| generated samples | 1 |
| valid samples | 0 |
| generated token count | 13 |
| decoded note count | 0 |
| passed generation gate | false |

Failure reason:

```text
generated MIDI has no notes
```

## Decision

Issue #18 validates the Stage B generation/decode plumbing, but not musical quality.

The first generated sample is invalid. That is still useful: the pipeline now catches the failure in a report instead of pretending a `.mid` file is enough.

## Next Step

Do not start broad jazz training yet.

Next issue should make the Stage B tiny-overfit probe stricter:

- train long enough to overfit a very small Stage B window set
- use deterministic or constrained token sampling for the first grammar check
- verify generated tokens contain complete `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups
- require at least one decoded MIDI sample to pass the review gate before expanding data size

Issue #20 implements the constrained grammar check. It produces complete note groups and non-zero decoded notes, but the full review gate still fails because overlapping notes exceed the current max-simultaneous-notes threshold.
