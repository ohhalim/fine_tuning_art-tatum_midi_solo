# Stage B Longer Phrase Probe

작성일: 2026-05-21

## Issue

- Issue: #49
- Branch: `issue-49-stage-b-longer-phrase-probe`
- Goal: 2-bar `coverage_chord` 후보가 너무 짧고 미완성처럼 느껴지는 문제를 4-bar phrase probe로 다시 검증한다.

## Why

Issue #45와 #47은 처음으로 reviewable `coverage_chord` MIDI 후보를 만들고 export했다.

하지만 manual piano-roll review에서 중요한 문제가 남았다.

- MIDI 파일은 valid하다.
- note count와 chord-tone ratio도 이전보다 낫다.
- melodic fragment처럼 보일 수는 있다.
- 그러나 2-bar 기준이라 phrase가 아니라 "만들다 만 단어"처럼 느껴진다.

따라서 이번 이슈의 기준은 "파일이 생성됐는가"가 아니다.

이번 기준:

- 최소 4마디 길이
- bar마다 충분한 onset coverage
- sample마다 32 note groups
- one-note/two-note/chord-block/long-sustain 실패 제외
- exported MIDI를 실제로 듣고 phrase sketch인지 판단 가능해야 함

## Implementation

Added or updated:

- `scripts/agent_harness.sh stage-b-longer-phrase-probe`
- `scripts/export_stage_b_review_candidates.py`
- `tests/test_stage_b_review_export.py`

The review exporter now accepts two report shapes:

- candidate ranking report with `top_candidates`
- generation probe report with `samples`

This lets the longer phrase probe export review candidates directly from:

```text
outputs/stage_b_generation_probe/harness_stage_b_longer_phrase_probe/report.json
```

Generated review package:

```text
outputs/stage_b_review_candidates/harness_stage_b_longer_phrase_probe/review_manifest.json
outputs/stage_b_review_candidates/harness_stage_b_longer_phrase_probe/review_candidates.md
outputs/stage_b_review_candidates/harness_stage_b_longer_phrase_probe/midi/*.mid
```

Generated artifacts are local outputs and are not committed.

## Harness Command

```bash
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```

Equivalent generation setup:

```bash
./.venv/bin/python scripts/run_stage_b_generation_probe.py \
  --run_id harness_stage_b_longer_phrase_probe \
  --issue_number 49 \
  --max_files 2 \
  --window_bars 4 \
  --window_stride_bars 2 \
  --min_window_target_notes 8 \
  --epochs 3 \
  --batch_size 8 \
  --max_sequence 192 \
  --num_samples 3 \
  --bars 4 \
  --generation_mode constrained \
  --coverage_aware_positions \
  --coverage_position_window 0 \
  --chord_aware_pitches \
  --chord_pitch_mode tones \
  --chord_pitch_repeat_window 2 \
  --constrained_note_groups_per_bar 8 \
  --postprocess_overlap \
  --max_simultaneous_notes 2 \
  --temperature 0.9 \
  --top_k 2 \
  --min_valid_samples 1 \
  --min_strict_valid_samples 1 \
  --max_collapse_warning_sample_rate 0.34 \
  --require_all_grammar_samples \
  --require_note_groups \
  --n_layers 1 \
  --num_heads 4 \
  --d_model 64 \
  --dim_feedforward 128 \
  --lora_r 4 \
  --lora_alpha 8
```

## Probe Result

Local manual probe result before this doc:

| Metric | Value |
|---|---:|
| generated samples | 3 |
| strict valid samples | 3 |
| grammar valid samples | 3 |
| bars | 4 |
| note groups per bar | 8 |
| note groups per sample | 32 |
| avg onset coverage ratio | 0.500 |
| avg sustained coverage ratio | 0.682 |
| avg position span ratio | 0.969 |
| max longest sustained empty run | 2 |
| collapse warning samples | 0 |
| repeated pitch ratio | 0.719 |

Sample 1 metrics:

| Metric | Value |
|---|---:|
| note count | 32 |
| unique pitch count | 9 |
| chord-tone ratio | 0.906 |
| phrase coverage ratio | 0.984 |
| dead-air ratio | 0.484 |
| max simultaneous notes | 2 |

## Review Checklist

Open the exported 4-bar MIDI candidates and check:

- Does it feel like a phrase, not a fragment?
- Is there a beginning, continuation, and landing?
- Does the line avoid one repeated pitch as the main idea?
- Is the high repeated-pitch ratio musically acceptable as motif, or just mechanical reuse?
- Does chord-tone correctness sound too constrained or robotic?
- Is the rhythm still too grid/mechanical?
- Does the high register bias make it unusable as a jazz piano solo?

## Current Interpretation

This probe fixes the length problem structurally, but it does not prove musical phrase quality.

The important remaining risk is repeated-pitch dependence:

- note count is now high enough for review
- bar coverage is regular and complete
- chord-tone ratio is high
- repeated pitch ratio is also high, around `0.719`

So the review question changed from:

> "Is this too short to judge?"

to:

> "Does this 4-bar line sound like a motif/phrase, or just a mechanically repeated pitch pattern?"

## Decision Boundary

If the 4-bar candidates sound like usable phrase sketches:

- move to generic jazz base training design
- keep Brad subset for later adaptation/holdout
- avoid claiming Brad style yet

If the 4-bar candidates still sound like fragments:

- do not start broad training
- add phrase/motif-level constraints or evaluate a pretrained symbolic MIDI base
- do not keep adding postprocess and call it model quality

## Validation

Required validation:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/export_stage_b_review_candidates.py tests/test_stage_b_review_export.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```
