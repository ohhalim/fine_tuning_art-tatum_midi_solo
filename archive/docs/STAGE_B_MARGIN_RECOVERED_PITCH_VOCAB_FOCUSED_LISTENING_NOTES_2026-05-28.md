# Stage B Margin-Recovered Pitch Vocabulary Focused Listening Notes

작성일: 2026-05-28

## 목적

Issue #258에서 `keep_for_focused_listening`으로 분리된 pitch vocabulary 후보를 focused listening review notes template로 넘긴다.

이 단계는 실제 청감 판단이 아니라, 다음 review fill에서 timing / chord fit / phrase continuation / landing / jazz vocabulary를 일관되게 기록하기 위한 준비다.

## 입력

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4` |
| prior decision | `keep_for_focused_listening` |
| focused context source | Issue #258 focused context decision |

## 구현

- `scripts/build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py`
  - focused package와 focused context decision을 함께 읽음
  - 기존 focused listening notes template builder 재사용
  - prior decision을 `keep_for_focused_listening`으로 보존
  - focused context metrics와 review risks를 notes candidate에 추가
- `stage-b-margin-recovered-pitch-vocab-focused-listening-notes` harness
  - focused package / context decision이 없으면 Issue #258 harness 선행 실행
  - notes template과 summary 생성

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_pitch_vocab_focused_listening_notes
bash scripts/agent_harness.sh stage-b-margin-recovered-pitch-vocab-focused-listening-notes
```

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| review risks | `dead_air_ratio_at_gate`, `adjacent_pitch_repeats` |

Focused context metrics carried into notes:

| 항목 | 값 |
|---|---:|
| note count | `13` |
| unique pitch count | `6` |
| phrase span | `6.250` beats |
| dead-air ratio | `0.400` |
| onset coverage | `0.500` |
| sustained coverage | `0.625` |
| adjacent pitch repeats | `3` |
| duplicated 3-note chunks | `0` |
| max active notes | `1` |
| final note | `G#4` over `Fm7`, chord tone |

Pending fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- final decision

## 해석

- focused listening review 준비가 완료됐다.
- prior decision은 context metric 기준 `keep_for_focused_listening`이다.
- 실제 listening decision은 아직 `pending`이다.
- dead-air와 adjacent repeats는 다음 review fill의 명시적 risk로 유지된다.

## 다음 작업

다음 issue는 focused listening review fill이다.

판단 기준:

- timing이 dead-air `0.400` 때문에 끊기게 들리는지
- adjacent repeats `3`이 motif인지 기계적 반복인지
- final landing `G#4` over `Fm7`가 phrase closure로 충분한지
- keep / needs_followup / reject 중 하나로 분리
