# Stage B Margin-Recovered Pitch Vocabulary Focused Context Review

작성일: 2026-05-28

## 목적

Issue #256에서 pitch vocabulary hard gate를 통과한 후보를 focused solo/context review package로 격리한다.

확인 대상:

- context MIDI 생성 가능 여부
- solo-line max active `1` 유지 여부
- pitch vocabulary gate 통과 후보가 context 위에서도 blocker 없이 유지되는지
- dead-air `0.400`과 adjacent repeats `3`이 다음 review risk로 남는지

## 입력 후보

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4` |
| source run | `harness_stage_b_margin_recovered_pitch_vocab_seed17_topk5_temp090_n24` |
| sample | `4` |
| sample seed | `20` |
| previous decision | `pitch_vocab_qualified` |

## 구현

- `scripts/build_stage_b_margin_recovered_pitch_vocab_focused_package.py`
  - pitch vocabulary sweep summary의 selected candidate를 focused review notes 형태로 변환
  - 기존 margin-recovered focused package builder 재사용
  - solo-line MIDI와 chord/bass context MIDI 생성
- `stage-b-margin-recovered-pitch-vocab-focused-context` harness
  - sweep summary가 없으면 pitch vocab sweep 먼저 실행
  - selected candidate focused package 생성
  - focused context decision 실행

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_pitch_vocab_focused_package
bash scripts/agent_harness.sh stage-b-margin-recovered-pitch-vocab-focused-context
```

## 결과

Focused package:

| 항목 | 값 |
|---|---:|
| candidate count | `1` |
| copied MIDI files | `2` |
| source note count | `16` |
| focused note count | `13` |
| removed notes | `3` |
| source max active | `2` |
| focused max active | `1` |
| context bars | `2` |

Focused context decision:

| 항목 | 값 |
|---|---|
| decision | `keep_for_focused_listening` |
| decision flags | `[]` |
| note count | `13` |
| unique pitch count | `6` |
| range | `D#4-C5` |
| phrase span | `6.250` beats |
| max active notes | `1` |
| dead-air ratio | `0.400` |
| onset coverage | `0.500` |
| sustained coverage | `0.625` |
| adjacent pitch repeats | `3` |
| duplicated 3-note chunks | `0` |
| final note | `G#4` over `Fm7`, chord tone |
| context tracks | chord guide / bass guide / solo present |

## 해석

- selected candidate는 focused context metric gate를 통과했다.
- context MIDI에는 chord guide, bass guide, solo track이 모두 있다.
- solo-line postprocess 후 max active notes는 `1`이다.
- final landing은 `G#4` over `Fm7` chord tone이다.
- 하지만 dead-air는 gate 상한 `0.400`에 붙어 있고, adjacent pitch repeats는 `3`이다.
- 이 결과는 focused listening notes로 올릴 수 있다는 의미이며, 실제 human/audio listening preference나 broad model quality 증명은 아니다.

## 다음 작업

다음 issue는 focused listening review notes 생성이다.

기록해야 할 risk:

- dead-air `0.400`
- adjacent pitch repeats `3`
- phrase span `6.250` beats
- final landing chord-tone 여부
- timing / phrase continuation / jazz vocabulary는 pending 상태로 유지
