# Dataset Strategy

작성일: 2026-05-18

## Decision

Brad Mehldau style model을 바로 학습하지 않는다.

먼저 전체 jazz piano MIDI corpus로 generic jazz pianist base를 만들 수 있는지 확인하고, 그 다음 Brad Mehldau subset으로 style adaptation을 검토한다.

```text
all jazz piano MIDI
  -> quality audit
  -> filtered non-Brad generic jazz-piano training set
  -> generic jazz pianist base
  -> Brad Mehldau adaptation
  -> Brad holdout evaluation
```

## Why

Brad Mehldau 데이터만으로 from-scratch model을 만들기에는 데이터가 작다.

현재 파일 시스템 기준:

| Split | MIDI files |
|---|---:|
| physical MIDI paths under `midi_dataset` | 5554 |
| active audit tree: `midi_dataset/midi` | 2777 |
| duplicate mirror tree: `midi_dataset/midi_kong` | 2777 |
| active studio | 1994 |
| active live | 783 |
| Brad Mehldau studio | 18 |
| Brad Mehldau live | 54 |
| Brad Mehldau total | 72 |

따라서 Brad 데이터는 최종 style adaptation과 holdout evaluation에 더 적합하다.

The active training/audit tree is `midi_dataset/midi`. Do not train both `midi/` and `midi_kong/` as if they were independent data.

## Risk

전체 dataset을 바로 학습에 넣으면 안 된다.

확인해야 할 위험:

- unreadable/corrupt MIDI
- too-short MIDI
- very long MIDI
- non-piano or multi-instrument MIDI
- score-like files vs performance-like files
- long sustain artifacts
- duplicate files
- skewed artist distribution
- live/studio quality mismatch
- token length outliers

이 문제가 해결되지 않으면 모델은 jazz piano grammar가 아니라 dataset artifact를 배울 수 있다.

## Audit Command

```bash
python scripts/audit_jazz_piano_dataset.py
```

Outputs:

```text
outputs/dataset_audit/jazz_piano_dataset_audit.json
outputs/dataset_audit/jazz_piano_dataset_audit.md
```

These outputs are generated artifacts and should not be committed.

Current full audit result for `midi_dataset/midi`:

| Metric | Value |
|---|---:|
| files | 2777 |
| readable | 2777 |
| unreadable | 0 |
| candidate | 2775 |
| candidate non-Brad | 2703 |
| candidate Brad | 72 |
| review too long | 1 |
| reject too few notes | 1 |
| exact duplicate hash groups | 0 |

Important distribution stats:

| Metric | P50 | P90 | P99 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| duration sec | 271.23 | 436.70 | 682.72 | 2327.62 | 286.60 |
| non-drum note count | 2068 | 3838 | 6556 | 17446 | 2280.68 |
| max note duration ratio | 0.02 | 0.06 | 0.13 | 0.77 | 0.03 |

Top artists by file count:

| Artist | Files |
|---|---:|
| Dick Hyman | 284 |
| Art Tatum | 122 |
| Abdullah Ibrahim | 84 |
| Dave McKenna | 76 |
| Chick Corea | 73 |
| Brad Mehldau | 72 |
| Fred Hersch | 66 |
| Ray Bryant | 63 |
| Oscar Peterson | 56 |

Interpretation:

- The active tree is clean enough to be a generic jazz-piano base candidate.
- The corpus is piano-like: one instrument per file and piano program ratio `1.0` in the audit.
- Brad is a small but usable style-adaptation subset.
- The current bottleneck is more likely tokenization/training design than raw dataset readability.

Fast smoke:

```bash
python scripts/audit_jazz_piano_dataset.py --max_files 100
```

## Audit Decisions

The audit labels files as:

- `candidate`
- `reject_unreadable`
- `reject_too_few_notes`
- `reject_too_short`
- `review_pitch_range`
- `review_long_sustain`
- `review_non_piano_program`
- `review_too_long`

Only `candidate` files should enter the first training manifest.

Review files require a reasoned decision before training.

## Training Splits

After audit, build manifests conceptually like this:

```text
candidate_non_brad
  -> generic_jazz_train
  -> generic_jazz_val

candidate_brad
  -> brad_adaptation_train
  -> brad_adaptation_val
  -> brad_test_holdout
```

Rules:

- exact duplicate files must not cross train/val/test boundaries
- Brad files should not be used in the generic non-Brad baseline
- Brad holdout must stay unseen until evaluation
- live and studio Brad files may need separate reporting

## Current Model Order

1. Run full dataset audit. Completed for `midi_dataset/midi`.
2. Keep `control_v1` 2-file probe as a pipeline sanity check.
3. If `control_v1` still makes sustain/chord blocks, move to Stage B tokenization before broad training.
4. Train generic jazz pianist base only after tokenizer sanity is proven.
5. Fine-tune/adapt on Brad only after generic base outputs reviewable MIDI.

## Stage B Implication

If Stage B is needed, the full dataset should be rebuilt as phrase/window records with explicit:

- bar
- position
- chord
- pitch
- duration
- velocity
- tempo

The whole-dataset strategy does not remove the need for better tokenization.

It only changes the training objective:

```text
not: Brad-only from scratch
yes: generic jazz piano prior -> Brad style adaptation
```
