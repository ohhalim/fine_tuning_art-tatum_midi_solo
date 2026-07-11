# Stage B Margin-Recovered Timing/Repetition Focused Listening Fill

## 요약

Issue #270은 Issue #268 pending focused listening notes를 MIDI/context evidence 기준으로 채운 작업이다.

Issue #264/#266에서 objective timing과 context blocker는 개선됐지만, 실제 listening decision에서는 phrase continuation과 vocabulary risk를 다시 판단해야 했다.

## 변경

- timing/repetition focused listening fill script 추가
- pending notes를 reviewed 상태로 채움
- timing, chord fit, phrase continuation, landing, jazz vocabulary, decision 기록
- evidence에 dead-air, adjacent repeat, max interval, final landing, review risks 보존

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39` |
| reviewed count | `1` |
| pending count | `0` |
| prior decision | `keep_for_focused_listening` |
| final decision | `needs_followup` |
| timing | `acceptable` |
| chord fit | `acceptable` |
| phrase continuation | `weak` |
| landing | `acceptable` |
| jazz vocabulary | `thin` |
| dead-air ratio | `0.353` |
| adjacent pitch repeats | `2` |
| max interval | `16` |
| final landing | `A#4` over `Fm7`, tension |

## 해석

- Issue #262 대비 dead-air가 내려가 timing은 `stiff`에서 `acceptable`로 개선됐다.
- final landing은 outside가 아니라 tension이므로 landing blocker는 아니다.
- adjacent repeats와 wide interval 때문에 phrase continuation은 `weak`, jazz vocabulary는 `thin`으로 남았다.
- 따라서 focused listening fill 기준 최종 decision은 `needs_followup`이다.
- 이 결과는 broad model quality나 human/audio preference proof가 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_timing_repetition_focused_listening_fill
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-listening-fill
```

## 산출물

- script: `scripts/fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py`
- test: `tests/test_stage_b_margin_recovered_timing_repetition_focused_listening_fill.py`
- filled notes: `outputs/stage_b_margin_recovered_timing_repetition_focused_listening_fill/harness_stage_b_margin_recovered_timing_repetition_focused_listening_fill/focused_listening_review_notes_filled.json`
- markdown: `outputs/stage_b_margin_recovered_timing_repetition_focused_listening_fill/harness_stage_b_margin_recovered_timing_repetition_focused_listening_fill/focused_listening_review_notes_filled.md`
