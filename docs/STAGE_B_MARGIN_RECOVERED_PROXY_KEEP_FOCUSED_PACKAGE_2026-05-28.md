# Stage B Margin-Recovered Proxy Keep Focused Package

작성일: 2026-05-28

## 목적

Issue #248은 Issue #246에서 정리한 margin-recovered proxy keep 후보를 focused context review package로 격리한 작업이다.

중요한 경계:

- 이 패키지는 focused context review 입력이다.
- proxy `keep`은 final musical quality가 아니다.
- copied MIDI files는 `outputs/` artifact로만 두고 커밋하지 않는다.

## 입력

Source artifacts:

- proxy-filled review notes:
  - `outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.json`
- source generated MIDI:
  - `outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/samples/stage_b_sample_5.mid`
- source generation report:
  - `outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/report.json`

Selected candidate:

- `margin_recovered_rank_2_seed_31_sample_5`

## 구현

| 단계 | 내용 |
|---|---|
| Candidate filter | proxy review decision `keep` 후보만 선택 |
| Solo-line review MIDI | source generated MIDI를 simultaneous limit `1`로 focused solo-line 변환 |
| Context MIDI | generation request의 BPM/chord progression으로 chord guide context MIDI 생성 |
| Focused package | solo-line MIDI, context MIDI, source metrics, first-note summary를 package JSON/Markdown으로 기록 |

## Outputs

Generated package:

- package JSON:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package/focused_review_package.json`
- package markdown:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package/focused_review_package.md`
- copied solo-line MIDI:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package/midi/margin_recovered_rank_2_seed_31_sample_5_solo_line.mid`
- copied context MIDI:
  - `outputs/stage_b_margin_recovered_focused_package/harness_stage_b_margin_recovered_proxy_keep_focused_package/context_midi/margin_recovered_rank_2_seed_31_sample_5_with_context.mid`

## Result

Focused package summary:

| field | value |
|---|---|
| candidate count | `1` |
| copied MIDI files | `2` |
| candidate | `margin_recovered_rank_2_seed_31_sample_5` |
| decision | `keep` |
| timing | `acceptable` |
| phrase | `strong` |
| chord fit | `not_scored` |
| context chords | `Cm7`, `Fm7`, `Bb7`, `Ebmaj7` |
| context BPM | `124` |
| context bars | `2` |

Focused solo-line metrics:

| metric | value |
|---|---:|
| original note count | `19` |
| focused note count | `14` |
| unique pitch count | `4` |
| original max simultaneous notes | `2` |
| focused max simultaneous notes | `1` |
| focused postprocess removed notes | `5` |
| focused postprocess removal ratio | `0.263` |
| dead-air ratio | `0.444` |
| phrase coverage ratio | `0.937` |
| onset coverage ratio | `0.500` |
| sustained coverage ratio | `0.719` |

## 판단

Issue #248은 margin-recovered proxy keep 후보를 focused context review 가능한 단일 package로 격리했다.

이 단계에서 확보한 것:

- proxy keep 후보 1개만 package로 추출
- source generated MIDI와 focused solo-line MIDI의 차이 기록
- max simultaneous notes `1`인 focused solo-line review artifact 생성
- request chord progression 기반 context MIDI 생성
- 다음 focused context decision의 입력 고정

남은 위험:

- chord fit은 아직 `not_scored`다.
- 이 결과는 MIDI metric proxy와 package 생성 결과이며, human listening proof가 아니다.
- broad trained-model quality와 Brad style adaptation은 아직 미검증이다.

## 다음 작업

`Stage B margin-recovered focused context decision`

목표:

- focused solo/context MIDI를 기준으로 proxy keep을 유지할지 판단한다.
- register, phrase continuation, max interval, chord guide fit, dead-air 체감을 확인한다.
- focused context에서도 유지되면 listening review notes template로 넘긴다.

## 검증

실행한 검증:

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_focused_package
bash scripts/agent_harness.sh stage-b-margin-recovered-proxy-keep-focused-package
```
