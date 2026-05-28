# Stage B Model-Core MVP Completion Audit

작성일: 2026-05-28

## 결론

| 항목 | 판정 |
|---|---|
| 이 작업이 core인지 | 맞음 |
| model-core pipeline MVP | 조건부 완료 |
| unconstrained trained-model MVP | 미완료 |
| portfolio claim 가능 범위 | symbolic MIDI 생성 검증 파이프라인 |
| portfolio claim 제외 범위 | 완성된 개인화 재즈 생성 모델 |

## MVP 기준

현재 MVP는 제품 MVP가 아니라 model-core MVP다.

완료로 볼 수 있는 범위:

- MIDI dataset audit
- Stage B window/token dataset preparation
- tiny training path
- token-to-MIDI decode path
- constrained generation
- overlap/postprocess gate
- objective MIDI review
- focused candidate review package

완료로 볼 수 없는 범위:

- unconstrained model generation 품질
- broad multi-seed model quality
- Brad style adaptation
- human/audio preference 검증
- realtime DAW/plugin integration

## 검증 결과

| Harness | Run ID | 결과 | 핵심 수치 |
|---|---|---|---|
| `stage-b-window-prepare` | `issue_220_stage_b_window_prepare` | PASS | samples `70`, train/val `63/7`, vocab size `547`, max token id `544`, fits vocab `true` |
| `stage-b-generation-probe` | `issue_220_stage_b_generation_probe` | FAIL | train loss `6.2115`, val loss `5.9441`, valid samples `0/1`, failure `note count too low: 3 < 6` |
| `stage-b-overlap-gate` | `issue_220_stage_b_overlap_gate` | PASS | valid `1/1`, strict `1/1`, grammar `1/1`, note count `6`, unique pitches `5`, phrase coverage `0.9375` |
| `stage-b-rhythm-phrase-variation` | `issue_220_stage_b_rhythm_phrase_variation` | PASS | selected modes gate `true`, clean objective candidates `6/6`, duplicate note sequences `0` |

## Raw Generation 실패 원인

`stage-b-generation-probe`는 training path 자체는 실행됐지만, unconstrained raw generation sample이 review gate를 통과하지 못했다.

관찰값:

- note count: `3`
- minimum required note count: `6`
- complete note groups: `0`
- invalid token count: `78`
- valid sample count: `0/1`
- grammar gate sample count: `0/1`
- strict review gate: `false`

해석:

- 모델 학습/체크포인트 저장/샘플 생성 실행 경로는 동작한다.
- unconstrained token sampling은 아직 Stage B grammar를 안정적으로 유지하지 못한다.
- 따라서 "학습된 모델이 바로 좋은 MIDI를 생성한다"는 주장은 불가하다.

## Constrained Pipeline 통과 근거

`stage-b-overlap-gate`는 constrained generation과 overlap postprocess를 포함한 review gate를 통과했다.

관찰값:

- valid sample count: `1/1`
- strict valid sample count: `1/1`
- grammar gate sample count: `1/1`
- complete note groups: `8`
- invalid token count: `0`
- postprocess before/after note count: `8 -> 6`
- max simultaneous notes after postprocess: `2`
- phrase coverage ratio: `0.9375`

해석:

- `.mid` 파일 존재가 아니라 objective gate 기준으로 후보를 검증한다.
- constrained/postprocessed path에서는 reviewable solo-line 후보 생성이 가능하다.
- 이 범위까지는 model-core pipeline MVP로 볼 수 있다.

## Focused Candidate 경로

`stage-b-rhythm-phrase-variation`은 현재 focused candidate review 경로의 반복 가능성을 확인했다.

요약:

- selected modes gate: `true`
- compare gate: `true`
- `data_motif_contour_landing_repair`: valid `3/3`, strict `3/3`, final landing resolved `3/3`
- `data_motif_rhythm_phrase_variation`: valid `3/3`, strict `3/3`, final landing resolved `3/3`
- objective MIDI review: candidates `6`, clean `6`, flags `{}`
- review manifest: unique note sequences `6`, duplicate note sequences `0`
- listening review: pending `6`, reviewed `0`

해석:

- objective-clean candidate 생성과 review package 구성은 가능하다.
- 사람이 들은 청감 선호까지 검증한 결과는 아니다.
- listening review가 비어 있으므로 이 결과만으로 generation rule을 더 바꾸면 안 된다.

## 현재 판정

| 질문 | 답 |
|---|---|
| 이게 코어인가? | 맞음. dataset, tokenization, training, generation, decode, review gate가 연결된 model-core 작업이다. |
| MVP가 끝났는가? | pipeline MVP는 조건부 완료. unconstrained trained-model MVP는 미완료. |
| 지금 MIDI 생성이 되는가? | constrained/focused pipeline으로 생성 가능. raw trained model generation은 gate 미통과. |
| 머신러닝 모델이 만든다고 말할 수 있는가? | tiny training 경로는 있지만, 현재 통과 후보의 품질은 constraint/postprocess/review gate 의존성이 크다. |

## 다음 작업

다음 이슈는 unconstrained ML generation gate repair가 맞다.

목표:

- invalid token family 감소
- complete note group 생성률 개선
- note count gate 통과
- grammar gate 통과
- constrained path 의존도와 raw model 기여도 분리

권장 이슈명:

- `Stage B unconstrained model generation gate repair`

성공 기준:

- `stage-b-generation-probe` valid sample count `1+`
- grammar gate sample count `1+`
- complete note groups `6+`
- invalid token count 감소
- strict review gate 통과 후보 `1+`
