# Stage B Margin-Recovered Pitch Vocabulary Focused Listening Fill

작성일: 2026-05-28

## 목적

Issue #260 focused listening notes template의 pending fields를 MIDI/context evidence 기준으로 채운다.

이 작업은 human/audio listening proof가 아니라, 기존 metric과 context decision을 근거로 다음 repair 방향을 분리하는 proxy fill이다.

## 입력

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4` |
| prior decision | `keep_for_focused_listening` |
| review risks | `dead_air_ratio_at_gate`, `adjacent_pitch_repeats` |

## 구현

- `scripts/fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py`
  - focused listening notes template를 읽어 reviewed state로 채움
  - dead-air, adjacent repeat, phrase span, final landing evidence를 보존
  - timing / phrase continuation / jazz vocabulary risk를 decision에 반영
- `stage-b-margin-recovered-pitch-vocab-focused-listening-fill` harness
  - notes template이 없으면 Issue #260 harness 선행 실행
  - expected decision `needs_followup` 검증

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_pitch_vocab_focused_listening_fill
bash scripts/agent_harness.sh stage-b-margin-recovered-pitch-vocab-focused-listening-fill
```

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `1` |
| reviewed count | `1` |
| pending count | `0` |
| prior decision | `keep_for_focused_listening` |
| final decision | `needs_followup` |
| timing | `stiff` |
| chord fit | `strong` |
| phrase continuation | `weak` |
| landing | `strong` |
| jazz vocabulary | `thin` |

Evidence:

| 항목 | 값 |
|---|---:|
| dead-air ratio | `0.400` |
| adjacent pitch repeats | `3` |
| phrase span | `6.250` beats |
| final note | `G#4` over `Fm7`, chord tone |

## 해석

- chord fit과 landing은 blocker가 아니다.
- timing은 dead-air가 gate 상한에 붙어 있어 `stiff`로 기록한다.
- phrase continuation은 span과 dead-air tradeoff 때문에 `weak`으로 둔다.
- jazz vocabulary는 adjacent repeats와 unique pitch 경계 때문에 `thin`으로 둔다.
- 따라서 selected pitch-vocab 후보는 final keep이 아니라 `needs_followup`이다.

## 다음 작업

다음 issue는 pitch vocabulary를 유지하면서 dead-air와 adjacent repeats를 줄이는 follow-up repair다.

유지 조건:

- focused unique pitch count `>= 6`
- focused max active notes `1`
- duplicated 3-note chunks `0`
- final landing chord tone 또는 tension

개선 목표:

- dead-air ratio `< 0.400`
- adjacent pitch repeats `< 3`
- phrase continuation `acceptable` 이상
