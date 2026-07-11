# Stage B Coverage-Aware Generation

작성일: 2026-05-20

## Issue

- Issue: #37
- Branch: `issue-37-stage-b-coverage-aware-generation`
- Goal: Stage B 2-file Brad probe에서 발생한 dead-air failure를 broad training 전에 constrained generation 수준에서 줄일 수 있는지 검증한다.

## Background

Issue #35의 temporal coverage diagnostics는 다음을 보여줬다.

- grammar gate: `3/3`
- basic valid: `0/3`
- strict valid: `0/3`
- collapse warning: `0/3`
- avg onset coverage ratio: `0.167`
- avg sustained coverage ratio: `0.417`
- avg position span ratio: `0.740`
- max longest sustained empty run: `11` steps

해석:

- Stage B grammar와 pitch/position collapse가 즉시 실패 원인은 아니었다.
- 실패 원인은 sparse onset과 긴 empty span이었다.
- broad generic training으로 넘어가기 전에, constrained generation의 `POSITION` 선택만 coverage-aware로 바꿔 효과를 확인해야 했다.

## Implementation

추가한 핵심 기능:

- `coverage_aware_position_tokens(group_index, note_groups_per_bar, position_window=0)`
- `generate_stage_b_constrained_tokens(..., coverage_aware_positions=True)`
- CLI flags:
  - `--coverage_aware_positions`
  - `--coverage_position_window`
- harness mode:
  - `bash scripts/agent_harness.sh stage-b-coverage-aware-probe`

의도적으로 하지 않은 것:

- pitch/duration/velocity를 rule-based로 덮어쓰기
- postprocess를 더 세게 해서 gate만 통과시키기
- broad dataset training 시작

이번 변경은 `POSITION` 후보만 coverage-aware로 제한하고, 나머지 token family는 기존 model logits sampling을 유지한다.

## Probe Setup

Command:

```bash
bash scripts/agent_harness.sh stage-b-coverage-aware-probe
```

Generated report:

```text
outputs/stage_b_generation_probe/harness_stage_b_coverage_aware_probe/report.json
```

Setup:

- input: `./midi_dataset/midi/studio/Brad Mehldau`
- max files: `2`
- generated Stage B windows: `137`
- train samples: `123`
- val samples: `14`
- max token id: `544`
- vocab size: `547`
- training: 3 epochs, full tiny model path, CPU
- best observed val loss: `4.0892`
- generation mode: `constrained`
- coverage-aware positions: `true`
- coverage position window: `0`
- constrained note groups per bar: `4`
- top_k: `2`
- temperature: `0.9`
- samples: `3`

## Result

Summary:

| Metric | Value |
|---|---:|
| sample count | 3 |
| grammar gate sample count | 3 |
| basic valid sample count | 3 |
| strict valid sample count | 3 |
| collapse warning sample count | 0 |
| passed generation gate | true |
| passed strict review gate | true |
| avg onset coverage ratio | 0.250 |
| avg sustained coverage ratio | 0.427 |
| avg position span ratio | 0.813 |
| max longest sustained empty run | 6 |
| avg repeated position/pitch pair ratio | 0.250 |
| max repeated position/pitch pair ratio | 0.375 |
| avg postprocess removal ratio | 0.000 |

Per-sample:

| Sample | Valid | Strict | Notes | Unique pitches | Onset coverage | Sustained coverage | Dead-air ratio | Chord-tone ratio |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | true | true | 8 | 3 | 0.250 | 0.469 | 0.429 | 0.500 |
| 2 | true | true | 8 | 3 | 0.250 | 0.438 | 0.429 | 0.375 |
| 3 | true | true | 8 | 3 | 0.250 | 0.375 | 0.429 | 0.375 |

## Comparison With Issue #35

| Metric | #35 temporal diagnostics | #37 coverage-aware |
|---|---:|---:|
| grammar gate | 3/3 | 3/3 |
| basic valid | 0/3 | 3/3 |
| strict valid | 0/3 | 3/3 |
| collapse warning | 0/3 | 0/3 |
| avg onset coverage ratio | 0.167 | 0.250 |
| avg sustained coverage ratio | 0.417 | 0.427 |
| avg position span ratio | 0.740 | 0.813 |
| max longest sustained empty run | 11 | 6 |

## Decision

Coverage-aware constrained `POSITION` selection is a useful local fix for the dead-air failure mode.

It proves:

- the 2-file Brad Stage B setup can produce grammar-valid, basic-valid, and strict-valid MIDI samples
- the previous failure was strongly tied to temporal sparsity
- temporal coverage should be part of future generation review reports

It does not prove:

- unconstrained model generation quality
- Brad Mehldau style adaptation quality
- generic jazz base readiness
- production-quality improvisation

## Next Boundary

Next issue should not jump directly to full broad training.

Recommended next issue:

- compare plain constrained vs coverage-aware generation in an explicit A/B sweep
- test `note_groups_per_bar` values such as `4`, `6`, and `8`
- measure pass-rate, coverage, repetition, and chord-tone behavior together
- keep generated MIDI artifacts local

If the A/B sweep remains stable, then move to generic jazz base preparation/training with Stage B windows.
