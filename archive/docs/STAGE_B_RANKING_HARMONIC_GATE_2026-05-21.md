# Stage B Ranking Harmonic Gate

작성일: 2026-05-21

## Issue

- Issue: #43
- Branch: `issue-43-stage-b-ranking-harmonic-gate`
- Goal: Stage B candidate ranking이 low chord-tone, repeated pitch, mechanical onset pattern MIDI를 좋은 후보로 올리지 않게 한다.

## Why

Issue #41 ranking은 coverage-aware A/B sweep의 generated MIDI candidates를 score로 정렬했다.

그 결과 top candidate는 다음 파일이었다.

```text
outputs/stage_b_coverage_ab_sweep/harness_stage_b_candidate_ranking_ab_sweep_coverage_g8_k2_t0p9/samples/stage_b_sample_1.mid
```

하지만 piano-roll review 결과 이 MIDI는 solo-line 후보로 보기 어려웠다.

Observed failure:

- note count는 `16`이었지만 unique pitch는 `3`뿐이었다.
- dominant pitches가 반복됐다.
- bar마다 onset template이 기계적으로 반복됐다.
- bar-level chord-tone coverage가 약한 구간이 있었다.
- strict-valid MIDI가 곧 musical candidate라는 뜻이 아니었다.

이 문제는 evaluator failure다.

파일이 존재하고 strict gate를 통과해도, 한두 pitch 반복이나 긴 sustain/chord block이면 reviewable MIDI로 취급하면 안 된다.

## Implementation

`scripts/rank_stage_b_candidates.py`가 각 candidate MIDI를 직접 읽어 추가 diagnostics를 계산한다.

Added diagnostics:

- `bar_chord_tone_ratio`
- `min_bar_chord_tone_ratio`
- `dominant_pitch_ratio`
- `repeated_pitch_ratio`
- `onset_template_repetition_ratio`

Added review flags:

- `low_chord_tone_ratio`
- `low_bar_chord_tone_ratio`
- `dominant_pitch_repetition`
- `low_pitch_variety`
- `repeated_onset_template`

Ranking now prefers candidates in this order:

1. reviewable candidate
2. strict-valid candidate
3. valid candidate
4. candidate without harmonic red flags
5. score

The score also penalizes:

- low chord-tone ratio
- low per-bar chord-tone ratio
- dominant pitch repetition
- repeated pitch ratio
- repeated onset template

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-candidate-ranking
```

Latest summary:

| Metric | Value |
|---|---:|
| candidates | 18 |
| valid candidates | 12 |
| strict candidates | 12 |
| viable unflagged candidates | 0 |
| flagged candidates | 18 |

The previous misleading top candidate is now flagged with:

- `low_chord_tone_ratio`
- `low_bar_chord_tone_ratio`
- `dominant_pitch_repetition`
- `low_pitch_variety`

## Decision

Current Stage B output is not yet a good listening candidate.

This issue improves the truthfulness of the ranking report. It does not improve the model itself.

The next issue should modify generation behavior, not just ranking.

Recommended next issue:

```text
Stage B chord-aware pitch constrained generation 추가
```

Target:

- preserve the coverage-aware `POSITION` improvement
- constrain or weight `NOTE_PITCH` choices by current bar chord
- allow chord tones plus limited tensions
- reduce repeated dominant pitch collapse
- require at least one viable unflagged generated sample before moving to generic jazz base training

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_candidate_ranking
./.venv/bin/python -m compileall scripts/rank_stage_b_candidates.py
bash scripts/agent_harness.sh quick
bash scripts/agent_harness.sh stage-b-candidate-ranking
```
