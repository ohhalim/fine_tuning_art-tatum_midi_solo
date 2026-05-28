# Stage B Margin-Recovered Proxy Keep Consolidation

작성일: 2026-05-28

## 목적

Issue #244의 MIDI metric proxy review 결과를 포트폴리오와 진행 계획에서 사용할 수 있는 형태로 정리한다.

이 문서는 새 생성 규칙이나 새 모델 품질을 주장하지 않는다.
목적은 `margin_recovered_rank_2_seed_31_sample_5` 후보가 왜 proxy keep으로 분리됐는지, 그리고 이 결과로 어디까지 말할 수 있는지 고정하는 것이다.

## 입력 근거

Source:

- `docs/STAGE_B_MARGIN_RECOVERED_PROXY_REVIEW_FILL_2026-05-28.md`
- `outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_summary.json`
- `outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.md`

Candidate set:

| candidate | selected by dead-air | score | decision |
|---|:---:|---:|---|
| `margin_recovered_rank_1_seed_23_sample_1` | true | `0.251` | needs_followup |
| `margin_recovered_rank_2_seed_31_sample_5` | false | `0.698` | keep |
| `margin_recovered_rank_3_seed_17_sample_3` | false | `0.564` | needs_followup |

## 구현된 판단 흐름

| 단계 | 구현 내용 |
|---|---|
| Candidate export | 6-file / 5-sample-per-seed run에서 seed별 best 후보 3개 추출 |
| Listening note template | timing, phrase, vocabulary, decision field를 pending 상태로 생성 |
| Proxy review fill | MIDI metric 기반으로 timing, phrase, vocabulary, decision 자동 채움 |
| Decision consolidation | dead-air 단일 기준 selected best와 proxy keep 후보 분리 |

## 문제

Dead-air ratio만 낮은 후보를 selected best로 고르면 phrase richness가 낮은 후보가 우선될 수 있다.

Issue #244에서 rank `1` 후보는 dead-air 기준 selected best였지만 다음 문제가 있었다.

- proxy score `0.251`
- timing `stiff`
- phrase `weak`
- vocabulary `thin`
- decision `needs_followup`

즉, dead-air gate는 필요하지만 단독 ranking 기준으로는 부족하다.

## 해결

Proxy review score에 다음 항목을 함께 반영했다.

- note count
- phrase coverage
- onset coverage
- sustained coverage
- dead-air ratio
- postprocess removal ratio
- seed-level failure state

그 결과 rank `2` 후보가 keep으로 분리됐다.

Rank `2` 후보의 의미:

- candidate: `margin_recovered_rank_2_seed_31_sample_5`
- proxy score: `0.698`
- timing: `acceptable`
- phrase: `strong`
- vocabulary: `acceptable`
- decision: `keep`

## 현재 주장 가능한 것

| 주장 | 상태 |
|---|---|
| `.mid` 생성 여부가 아니라 note-level metric으로 후보를 검증하는 파이프라인 | 가능 |
| dead-air selected best와 phrase-rich proxy keep 후보를 분리하는 review flow | 가능 |
| seed/sample repeatability와 candidate margin을 기록하는 local gate | 가능 |
| rank `2` seed `31` sample `5`의 MIDI metric proxy keep | 가능 |
| human listening preference | 불가 |
| broad trained-model quality | 불가 |
| Brad style adaptation 완료 | 불가 |

## 다음 경계

다음 작업은 proxy keep 후보를 focused solo/context review package로 격리하는 것이다.

새 generation repair를 바로 추가하지 않는다.
먼저 rank `2` 후보를 사람이 듣거나 piano roll로 비교할 수 있는 형태로 고정해야 한다.

Recommended next issue:

- `Stage B margin-recovered proxy keep focused package`
