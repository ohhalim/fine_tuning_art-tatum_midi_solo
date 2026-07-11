# Stage B 8-Bar Approach Phrase Probe

작성일: 2026-05-21

## Context

Manual review after Issue #55:

- 이전보다 확실히 좋아졌다.
- MIDI는 멜로디처럼 이어진다.
- 하지만 아직 jazz solo라기보다는 초등학교 음악, 또는 "떴다 떴다 비행기" 같은 단순 선율 느낌이다.
- 근음과 다이아토닉 코드 구성음을 누르면서 나열하는 chord-scale exercise처럼 들린다.
- tension이 수치상 들어갔을 수는 있지만, 귀에는 jazz vocabulary처럼 잘 느껴지지 않는다.
- 4-bar는 phrase가 시작하다 끝나는 느낌이라 더 긴 sample이 필요하다.

Issue #57은 이 피드백을 기준으로 8-bar phrase와 approach/passing-note policy를 검증한다.

## Goal

이번 목표는 "좋은 재즈 솔로"가 아니다.

목표:

- 8-bar로 더 긴 phrase 후보를 만든다.
- 기존 `tones`, `tones_tensions`와 새 `approach_tensions`를 비교한다.
- tension을 단순 허용음이 아니라 코드톤으로 향하는 approach/passing 역할로 제한한다.
- mode별 MIDI 파일명을 명확히 export해서 manual review를 쉽게 한다.

## Implementation

Added/changed:

- `chord_pitch_mode=approach_tensions`
- `chord_approach_pitch_classes()`
- `analyze_stage_b_approach_resolution()`
- `approach_resolution` fields in sample reports
- summary fields:
  - `avg_approach_candidate_ratio`
  - `avg_approach_resolution_ratio`
- named comparison MIDI export:
  - `compare_named_midi/01_approach_tensions_...mid`
  - `compare_named_midi/02_tones_...mid`
  - `compare_named_midi/03_tones_tensions_...mid`
- harness:

```bash
bash scripts/agent_harness.sh stage-b-8bar-approach-phrase
```

The `approach_tensions` mode uses a pair rule:

- first note in each position pair: non-chord approach/passing class near a non-root chord tone
- second note in each position pair: nearby non-root chord tone resolution

This is still a deterministic probe policy, not learned jazz language.

## Local Result

Report:

```text
outputs/stage_b_pitch_mode_compare/harness_stage_b_8bar_approach_phrase/pitch_mode_compare_report.json
outputs/stage_b_pitch_mode_compare/harness_stage_b_8bar_approach_phrase/pitch_mode_compare_report.md
```

Named review MIDI:

```text
outputs/stage_b_review_candidates/harness_stage_b_8bar_approach_phrase/compare_named_midi/
```

Files:

```text
01_approach_tensions_rank_01_sample_3.mid
01_approach_tensions_rank_02_sample_1.mid
01_approach_tensions_rank_03_sample_2.mid
02_tones_rank_01_sample_3.mid
02_tones_rank_02_sample_2.mid
02_tones_rank_03_sample_1.mid
03_tones_tensions_rank_01_sample_1.mid
03_tones_tensions_rank_02_sample_3.mid
03_tones_tensions_rank_03_sample_2.mid
```

Summary:

| pitch mode | strict | chord | root | tension | approach | resolved | sustained |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tones` | 3/3 | 0.616 | 0.260 | 0.000 | 0.000 | 0.000 | 0.917 |
| `tones_tensions` | 3/3 | 0.557 | 0.198 | 0.328 | 0.312 | 0.283 | 0.904 |
| `approach_tensions` | 3/3 | 0.419 | 0.000 | 0.161 | 0.500 | 1.000 | 0.917 |

## Interpretation

The 8-bar probe is structurally successful:

- all three modes produced strict-valid MIDI
- all three modes cover the 8-bar phrase
- named review files are available
- `approach_tensions` eliminated root-tone usage in this run
- `approach_tensions` produced measurable approach/resolution pairs

But this is not enough to claim jazz phrasing:

- `approach_tensions` is still a rule-imposed pair pattern
- repeated pitch-set behavior remains high
- the phrase may still sound like a mechanical exercise
- real jazz vocabulary requires rhythm/motif/swing/phrase memory, not only pitch-class filtering

Current diagnosis:

> We have moved from "broken MIDI" to "valid but beginner-like melodic exercise."

That is progress, but not the target.

## Manual Review Request

Listen to the named files in this order:

1. `02_tones_rank_01_sample_3.mid`
2. `03_tones_tensions_rank_01_sample_1.mid`
3. `01_approach_tensions_rank_01_sample_3.mid`

Review questions:

- Does 8-bar length feel less unfinished than 4-bar?
- Does `approach_tensions` sound more like jazz phrasing, or only like forced chromatic exercise?
- Is `tones_tensions` still too nursery-rhyme/simple?
- Which mode is the best starting point for the next musical constraint?

## Decision

Do not broad-train yet based only on this.

Next useful work:

- add rhythm/motif cells instead of pitch-only constraints
- add syncopation or swing-aware position templates
- compare generated line against real MIDI phrase statistics
- then decide whether generic jazz base training is justified

## Validation

```bash
./.venv/bin/python -m unittest tests.test_request_conditioning tests.test_stage_b_generation_probe tests.test_stage_b_pitch_mode_compare
./.venv/bin/python -m unittest tests.test_stage_b_generation_probe tests.test_stage_b_pitch_mode_compare tests.test_stage_b_review_export tests.test_stage_b_coverage_ab_sweep
./.venv/bin/python -m compileall inference/app/schemas.py scripts/run_stage_b_generation_probe.py scripts/run_stage_b_pitch_mode_compare.py tests/test_request_conditioning.py
bash scripts/agent_harness.sh stage-b-8bar-approach-phrase
```
