# Stage B Margin-Recovered Focused Context Decision

작성일: 2026-05-28

## 목적

Issue #250은 Issue #248 focused package의 단일 proxy `keep` 후보를 solo/context MIDI metric 기준으로 다시 판단한 작업이다.

Decision:

- candidate: `margin_recovered_rank_2_seed_31_sample_5`
- prior proxy decision: `keep`
- focused context decision: `needs_followup`
- next boundary: focused pitch-vocabulary / dead-air follow-up

## 입력

Focused package:

- `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package_context_decision/focused_review_package.json`

MIDI:

- solo-line:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package_context_decision/midi/margin_recovered_rank_2_seed_31_sample_5_solo_line.mid`
- context:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package_context_decision/context_midi/margin_recovered_rank_2_seed_31_sample_5_with_context.mid`

Context chord cycle:

```text
Cm7 | Fm7 | Bb7 | Ebmaj7
```

## 구현

| 단계 | 내용 |
|---|---|
| Focused package load | single proxy keep candidate만 입력으로 사용 |
| Solo MIDI analysis | note count, unique pitch, range, phrase span, max active, repeated cell 측정 |
| Context MIDI analysis | chord guide, bass guide, solo track 존재 여부 확인 |
| Final role analysis | final note와 context chord의 chord/tension/outside role 계산 |
| Decision gate | focused listening으로 넘길지 `needs_followup`으로 내릴지 결정 |

## Evidence

Focused solo/context metrics:

| metric | value |
|---|---:|
| note count | `14` |
| unique pitch count | `4` |
| range | `D#4-C5` |
| phrase span | `7.500` beats |
| max active notes | `1` |
| max interval | `7` |
| adjacent pitch repeats | `3` |
| duplicated 3-note pitch-class chunks | `0` |
| dead-air ratio | `0.444` |
| onset coverage ratio | `0.500` |
| sustained coverage ratio | `0.719` |

Final note:

| field | value |
|---|---|
| final note | `C5` |
| final start | `8.250` beats |
| final chord | `Bb7` |
| final role | `tension` |

Context track check:

| track requirement | result |
|---|---|
| chord guide | present |
| bass guide | present |
| solo track | present |
| request bars | `2` |
| generated context bars | `3` |

## Decision

Focused context decision:

| field | value |
|---|---|
| prior proxy decision | `keep` |
| focused context decision | `needs_followup` |
| ready for focused listening notes | `no` |
| ready for broad training | `no` |
| ready for style adaptation claim | `no` |

Decision flags:

- `low_pitch_variety`
- `dead_air_needs_review`

Rationale:

- unique pitch count가 `4`로 낮아 phrase vocabulary가 얇다.
- dead-air ratio가 `0.444`로 focused listening에 올리기 전 timing/space follow-up이 필요하다.
- max active notes `1`, context track 존재, final tension landing은 통과했지만 keep 승격 근거로 충분하지 않다.

## 판단

Issue #250은 rank `2` proxy keep 후보를 final candidate로 올리지 않는다.

이 단계에서 확보한 것:

- proxy keep 후보를 focused context에서 재검증하는 자동 decision path
- context guide under-coverage를 막기 위한 focused package context bars 보정
- `keep`에서 `needs_followup`으로 내려가는 근거 기록

남은 위험:

- low pitch variety
- dead-air / sparse timing feel
- chord fit은 아직 human listening으로 확인되지 않음
- broad trained-model quality와 Brad style adaptation은 미검증

## 다음 작업

`Stage B margin-recovered focused pitch-vocabulary dead-air follow-up`

목표:

- low pitch variety를 늘리면서 max active `1` 유지
- dead-air ratio를 낮추면서 repeated-cell을 재도입하지 않기
- rank `2` 후보를 직접 repair할지, margin-recovered 후보군에서 fallback 후보를 비교할지 분리

## 검증

실행한 검증:

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_focused_context_decision
bash scripts/agent_harness.sh stage-b-margin-recovered-focused-context-decision
```
