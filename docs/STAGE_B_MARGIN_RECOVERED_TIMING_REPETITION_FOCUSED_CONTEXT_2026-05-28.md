# Stage B Margin-Recovered Timing/Repetition Focused Context

## 요약

Issue #266은 Issue #264에서 선택한 timing/repetition repair 후보를 solo/context package로 격리하고, focused context 기준으로 다시 검증한 작업이다.

이 단계는 objective repair 후보를 바로 최종 후보로 올리지 않고, chord guide, bass guide, solo track이 포함된 context MIDI 안에서 solo-line 조건과 final landing을 다시 확인하기 위한 경계다.

## 변경

- timing/repetition repair summary를 입력으로 받는 focused package builder 추가
- selected candidate의 solo-line MIDI와 context MIDI 복사/생성
- 기존 focused context decision 로직으로 note count, unique pitch, phrase span, max active, repeated cell, final landing 검증
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39` |
| copied MIDI files | `2` |
| focused context decision | `keep_for_focused_listening` |
| decision flags | `{}` |
| note count | `14` |
| unique pitch count | `7` |
| range | `C#4-G5` |
| phrase span | `6.500` beats |
| max active notes | `1` |
| dead-air ratio | `0.353` |
| onset coverage | `0.500` |
| sustained coverage | `0.688` |
| adjacent pitch repeats | `2` |
| duplicated 3-note pitch-class chunks | `0` |
| final landing | `A#4` over `Fm7`, tension |
| context tracks | chord guide, bass root guide, solo |

## 해석

- focused context blocker는 발견되지 않았다.
- solo-line max active `1`, phrase span `6.5` beats, context guide 존재 조건을 만족했다.
- final note는 `Fm7` 위 tension으로 처리되어 outside landing blocker가 아니다.
- 이 결과는 focused listening review 진입 조건이지, human/audio preference나 broad model quality proof가 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_timing_repetition_focused_package
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-context
```

## 산출물

- script: `scripts/build_stage_b_margin_recovered_timing_repetition_focused_package.py`
- test: `tests/test_stage_b_margin_recovered_timing_repetition_focused_package.py`
- package: `outputs/stage_b_margin_recovered_timing_repetition_focused_package/harness_stage_b_margin_recovered_timing_repetition_focused_package/focused_review_package.json`
- decision: `outputs/stage_b_margin_recovered_focused_context_decision/harness_stage_b_margin_recovered_timing_repetition_focused_context_decision/focused_context_decision.json`
