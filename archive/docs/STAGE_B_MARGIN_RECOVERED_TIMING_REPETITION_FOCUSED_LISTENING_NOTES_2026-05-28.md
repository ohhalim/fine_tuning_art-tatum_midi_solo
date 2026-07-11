# Stage B Margin-Recovered Timing/Repetition Focused Listening Notes

## 요약

Issue #268은 Issue #266 focused context keep 후보를 focused listening review notes template으로 넘긴 작업이다.

이 단계는 context decision을 최종 keep으로 확정하지 않고, 실제 listening fill에서 판단할 field를 pending으로 남기는 구조다.

## 변경

- timing/repetition focused context decision 기반 listening notes builder 추가
- candidate metrics, context summary, prior decision, review risk를 notes에 보존
- JSON notes template과 Markdown 요약 생성
- harness mode 추가

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39` |
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| review risks | `dead_air_ratio_remaining`, `adjacent_pitch_repeats`, `wide_interval_review` |
| note count | `14` |
| unique pitch count | `7` |
| phrase span | `6.500` beats |
| dead-air ratio | `0.353` |
| adjacent pitch repeats | `2` |
| max interval | `16` |
| final landing | `A#4` over `Fm7`, tension |

## 해석

- context keep 후보는 listening review 입력으로 준비됐다.
- timing, chord fit, phrase continuation, landing, jazz vocabulary, final decision은 모두 pending이다.
- dead-air는 Issue #262보다 개선됐지만 아직 `0.35` 근처라 listening risk로 유지했다.
- adjacent repeats와 wide interval도 실제 청감 판단 전에 risk로 남겼다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_timing_repetition_focused_listening_notes
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-listening-notes
```

## 산출물

- script: `scripts/build_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py`
- test: `tests/test_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py`
- notes: `outputs/stage_b_margin_recovered_timing_repetition_focused_listening_notes/harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes/focused_listening_review_notes_template.json`
- markdown: `outputs/stage_b_margin_recovered_timing_repetition_focused_listening_notes/harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes/focused_listening_review_notes_template.md`
