# Stage B Margin-Recovered Focused Fallback Comparison

작성일: 2026-05-28

## 목적

Issue #252는 margin-recovered 후보 3개 전체를 focused solo/context metric 기준으로 비교해 fallback 후보가 있는지 확인한 작업이다.

Issue #250에서 rank `2` proxy keep 후보가 `needs_followup`으로 내려갔기 때문에, 바로 generation repair로 가기 전에 rank `1`, `3` 후보도 같은 기준으로 비교했다.

## 입력

Proxy-filled review notes:

- `outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.json`

All-candidate focused package:

- `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_focused_fallback_package/focused_review_package.json`

Focused fallback decision:

- `outputs/stage_b_margin_recovered_focused_context_decision/harness_stage_b_margin_recovered_focused_fallback_decision/focused_context_decision.json`

## 구현

| 단계 | 내용 |
|---|---|
| all-candidate package | focused package builder에서 decision `all` 지원 |
| context decision | 후보 3개 전체에 동일한 focused context metric 적용 |
| fallback comparison | proxy keep 후보와 needs_followup 후보를 같은 기준으로 비교 |
| blocker aggregation | 후보별 decision flag와 전체 flag count 기록 |

## Result

Focused fallback comparison:

| candidate | prior decision | focused decision | notes | unique | range | span | max active | dead-air | final | flags |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|
| `margin_recovered_rank_1_seed_23_sample_1` | needs_followup | needs_followup | `4` | `4` | `C5-E5` | `3.500` | `1` | `0.375` | `C5` over `Fm7`, chord tone | too_sparse, low_pitch_variety, short_phrase_span |
| `margin_recovered_rank_2_seed_31_sample_5` | keep | needs_followup | `14` | `4` | `D#4-C5` | `7.500` | `1` | `0.444` | `C5` over `Bb7`, tension | low_pitch_variety, dead_air_needs_review |
| `margin_recovered_rank_3_seed_17_sample_3` | needs_followup | needs_followup | `11` | `4` | `C#4-G#4` | `7.750` | `1` | `0.500` | `G#4` over `Fm7`, chord tone | too_sparse, low_pitch_variety, dead_air_needs_review |

Aggregate:

| field | value |
|---|---:|
| candidate count | `3` |
| focused `keep_for_focused_listening` | `0` |
| focused `needs_followup` | `3` |
| low pitch variety | `3` |
| dead-air needs review | `2` |
| too sparse for context review | `2` |
| short phrase span | `1` |

## 판단

Issue #252 결과, margin-recovered 후보군 안에는 focused listening으로 올릴 fallback 후보가 없다.

확인한 점:

- 모든 후보가 max active `1` solo-line artifact로 변환 가능하다.
- final landing은 chord tone 또는 tension으로 기록된다.
- 그러나 세 후보 모두 unique pitch count `4`로 낮다.
- rank `1`은 너무 짧고 sparse하다.
- rank `2`는 proxy keep이었지만 dead-air와 pitch variety가 남는다.
- rank `3`은 phrase span은 길지만 focused solo-line 변환 후 note count `11`, unique pitch `4`, dead-air `0.500`이다.

## 다음 작업

`Stage B margin-recovered pitch-vocabulary dead-air repair`

목표:

- unique pitch count를 늘린다.
- dead-air ratio를 낮춘다.
- max active `1`과 repeated-cell-free 조건을 유지한다.
- fallback 후보 선택으로 해결하려 하지 않고 generation/selection repair target을 분리한다.

## 검증

실행한 검증:

```bash
.venv/bin/python -m unittest tests.test_focused_review_package tests.test_stage_b_margin_recovered_focused_package tests.test_stage_b_margin_recovered_focused_context_decision
bash scripts/agent_harness.sh stage-b-margin-recovered-focused-fallback-comparison
```
