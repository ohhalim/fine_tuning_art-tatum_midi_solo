# Stage B Swing/Motif Phrase Grammar Probe

작성일: 2026-05-21

## Context

Manual piano-roll review after Issue #57 found that the 8-bar `approach_tensions` output is valid MIDI, but still not convincing jazz phrasing.

Observed problem:

- 이전보다 melodic line처럼 보인다.
- 하지만 근음/코드톤 중심의 단순 선율처럼 들린다.
- "떴다 떴다 비행기" 같은 beginner exercise 느낌이 남아 있다.
- 8-bar length와 approach/passing pitch만으로는 jazz vocabulary가 생기지 않는다.

Direct MIDI inspection showed the main rhythmic cause:

- baseline position pattern is almost fixed per bar.
- baseline IOI mostly alternates between `1` and `3` grid steps.
- baseline duration is dominated by one duration bucket.
- pitch가 바뀌어도 rhythm이 반복되므로 solo phrase보다 exercise처럼 들린다.

Issue #59 tests whether a swing/motif-oriented rhythm grammar reduces that mechanical feel.

## Goal

이번 목표는 좋은 jazz solo를 완성하는 것이 아니다.

목표:

- 8-bar `approach_tensions` pitch policy는 유지한다.
- `POSITION` 선택을 bar마다 바뀌는 syncopated motif template으로 제한한다.
- `NOTE_DURATION` 선택도 motif별로 다르게 제한한다.
- baseline과 swing/motif grammar를 같은 checkpoint에서 비교한다.
- MIDI review 파일명을 명확히 export한다.

## Implementation

Added/changed:

- `jazz_rhythm_position_tokens()`
- `jazz_rhythm_duration_tokens()`
- `analyze_stage_b_rhythm_profile()`
- generation flags:
  - `--jazz_rhythm_positions`
  - `--jazz_duration_tokens`
  - `--jazz_rhythm_profile swing_motif`
- rhythm summary fields:
  - `avg_syncopated_onset_ratio`
  - `avg_unique_bar_position_pattern_ratio`
  - `avg_duration_diversity_ratio`
  - `avg_most_common_duration_ratio`
  - `avg_ioi_diversity_ratio`
  - `avg_most_common_ioi_ratio`
- review ranking score now considers rhythm profile.
- new comparison runner:
  - `scripts/run_stage_b_phrase_grammar_compare.py`
- new harness:

```bash
bash scripts/agent_harness.sh stage-b-swing-motif-phrase
```

The new mode is:

```text
swing_motif_approach = approach_tensions pitch grammar + swing_motif position/duration grammar
```

This is still a constrained generation probe, not learned jazz language.

## Local Result

Report:

```text
outputs/stage_b_phrase_grammar_compare/harness_stage_b_swing_motif_phrase/phrase_grammar_compare_report.json
outputs/stage_b_phrase_grammar_compare/harness_stage_b_swing_motif_phrase/phrase_grammar_compare_report.md
```

Named review MIDI:

```text
outputs/stage_b_review_candidates/harness_stage_b_swing_motif_phrase/compare_named_midi/
```

Files:

```text
01_approach_baseline_rank_01_sample_3.mid
01_approach_baseline_rank_02_sample_1.mid
01_approach_baseline_rank_03_sample_2.mid
02_swing_motif_approach_rank_01_sample_1.mid
02_swing_motif_approach_rank_02_sample_3.mid
02_swing_motif_approach_rank_03_sample_2.mid
```

Summary:

| grammar | strict | root | tension | resolved | sync | bar-var | dur-rep | ioi-var | ioi-rep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `approach_baseline` | 3/3 | 0.000 | 0.161 | 1.000 | 0.500 | 0.125 | 0.552 | 0.032 | 0.508 |
| `swing_motif_approach` | 3/3 | 0.000 | 0.172 | 1.000 | 0.750 | 0.500 | 0.380 | 0.063 | 0.476 |

Comparison:

- syncopated onset ratio improved by `+0.250`.
- unique bar-position pattern ratio improved from `0.125` to `0.500`.
- most common duration ratio dropped from `0.552` to `0.380`.
- most common IOI ratio dropped from `0.508` to `0.476`.
- all samples passed strict review gate in both modes.

## Direct MIDI Structure Check

Top baseline file:

```text
01_approach_baseline_rank_01_sample_3.mid
notes: 62
unique pitches: 9
duration counts: 3-step duration dominates
IOI counts: mostly 1-step and 3-step alternation
bar position patterns: mostly [1, 4, 5, 8, 9, 12, 13, 16]
```

Top swing/motif file:

```text
02_swing_motif_approach_rank_01_sample_1.mid
notes: 63
unique pitches: 12
duration counts: 1-step, 2-step, and 3-step durations are more balanced
IOI counts: 1-step, 2-step, and 3-step gaps all appear
bar position patterns:
  bar 1: [0, 3, 5, 7, 10, 11, 13, 15]
  bar 2: [1, 3, 4, 7, 9, 12, 14, 15, 16]
  bar 3: [2, 5, 6, 8, 11, 13, 14]
  bar 4: [2, 4, 5, 8, 10, 12, 13, 15, 16]
```

This confirms that the previous nursery-rhyme/exercise feel was not only a pitch problem. The rhythm grid itself was too repetitive.

## Interpretation

This is a useful improvement, but not the final musical target.

What improved:

- onset placement is less mechanically repeated
- duration selection is less dominated by one bucket
- bar-to-bar rhythm template variation increased
- exported candidates are long enough to review as phrases

What did not improve enough:

- the pitch vocabulary is still small
- repeated pitch-set behavior remains high
- phrase still may sound like a rule-generated line
- this is not Brad style adaptation
- this is not evidence that the model learned jazz language

Current diagnosis:

> We reduced the mechanical rhythm-grid problem. The next bottleneck is still jazz vocabulary and motif development, not MIDI validity.

## Decision

Do not broad-train yet only from this result.

Next useful work:

- add motif-level pitch cells instead of only chord/tension classes
- compare generated rhythm profile against real jazz MIDI phrase windows
- add cadence/landing constraints for phrase endings
- then decide whether generic jazz base training is justified

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe tests.test_stage_b_phrase_grammar_compare tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/run_stage_b_generation_probe.py scripts/run_stage_b_phrase_grammar_compare.py scripts/run_stage_b_sampling_sweep.py scripts/export_stage_b_review_candidates.py tests/test_stage_b_generation_probe.py tests/test_stage_b_phrase_grammar_compare.py
bash scripts/agent_harness.sh stage-b-swing-motif-phrase
```
