# Stage B Raw Generation Gate Repair

작성일: 2026-05-28

## 결론

| 항목 | 판정 |
|---|---|
| raw Stage B generation gate | 통과 |
| model-core MVP | 로컬 tiny-overfit/review gate 기준 완료 |
| broad model quality | 미검증 |
| style adaptation | 미검증 |
| postprocess 의존성 | 있음 |

## 변경 사항

`stage-b-generation-probe` harness를 raw generation gate 검증용으로 조정했다.

변경 전:

- epochs `1`
- samples `1`
- top_k default `32`
- overlap postprocess 없음
- gate 실패여도 harness exit code `0`

변경 후:

- epochs `50`
- samples `5`
- top_k `4`
- overlap postprocess 사용
- note group / valid / strict gate 실패 시 harness 실패 처리

## 실패 분리

Issue #220 audit 기준 실패:

| 조건 | 결과 |
|---|---|
| epochs | `1` |
| top_k | `32` |
| valid samples | `0/1` |
| grammar samples | `0/1` |
| failure | `note count too low: 3 < 6` |
| complete note groups | `0` |
| invalid token count | `78` |

추가 확인:

| 조건 | 결과 | 해석 |
|---|---|---|
| epochs `1`, top_k `1` | invalid token count `47`, note count `1` | sampling 폭 문제가 아님 |
| epochs `50`, top_k `1` | grammar `3/3`, pitch `1`개 붕괴 | grammar 학습은 가능하지만 greedy collapse 발생 |
| epochs `50`, top_k `4` | grammar `5/5`, pitch 다양성 회복, max simultaneous `3-4` | 남은 실패는 polyphony gate |
| epochs `50`, top_k `4`, overlap postprocess | strict `5/5` | raw model output + postprocess gate 통과 |

## 최종 검증

Command:

```bash
RUN_ID=issue_222_stage_b_generation_probe bash scripts/agent_harness.sh stage-b-generation-probe
```

Run ID:

- `issue_222_stage_b_generation_probe`

Training:

- train samples: `63`
- val samples: `7`
- best validation loss: `1.6905`

Summary:

| 항목 | 값 |
|---|---:|
| sample count | `5` |
| valid sample count | `5` |
| strict valid sample count | `5` |
| grammar gate sample count | `5` |
| collapse warning sample count | `0` |
| passed generation gate | `true` |
| passed grammar gate | `true` |
| passed strict review gate | `true` |

Sample range:

| 항목 | 값 |
|---|---:|
| complete note groups | `21-22` |
| invalid token count | `0` |
| postprocess note count | `13-18` |
| unique pitch count | `4-6` |
| max simultaneous notes | `2` |
| phrase coverage ratio | `0.8125-1.0` |
| postprocess removal ratio | `0.1818-0.3810` |

## 해석

증명한 것:

- Stage B raw token generation이 충분한 tiny-overfit 조건에서 grammar-valid sequence를 만들 수 있음
- top_k `4` sampling이 greedy single-pitch collapse를 줄임
- overlap postprocess 후 MIDI review gate와 strict collapse gate 통과 가능
- `stage-b-generation-probe`가 실패를 실제 exit code로 드러내는 검증 하네스가 됨

아직 증명하지 않은 것:

- broad multi-file/multi-seed quality
- 사람이 듣는 음악적 선호
- Brad Mehldau style adaptation
- postprocess 없이 solo-line polyphony gate 통과
- 장기 phrase 품질

## 다음 작업

권장 다음 이슈:

- `Stage B raw generation broader repeatability sweep`

성공 기준:

- 2개 이상 source file
- 3개 이상 seed
- raw + postprocess strict pass-rate 기록
- postprocess removal ratio 분포 기록
- listening review 후보와 non-reviewable 후보 분리
