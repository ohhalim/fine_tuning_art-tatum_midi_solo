# Stage B Margin-Recovered MIDI Proxy Review Fill

작성일: 2026-05-28

## 결론

Issue #242의 pending listening review notes를 MIDI metric 기반 proxy review로 채웠다.
실제 청감 review가 아니라, dead-air, note count, phrase/onset/sustained coverage, postprocess removal, seed failure 상태를 기준으로 한 proxy 판단이다.

| 항목 | 결과 |
|---|---:|
| candidate count | `3` |
| reviewed count | `3` |
| keep | `1` |
| needs_followup | `2` |
| reject | `0` |
| proxy keep candidate | `margin_recovered_rank_2_seed_31_sample_5` |
| human listening proof | false |

dead-air 기준 selected best였던 rank `1`은 proxy review에서 `needs_followup`으로 내려갔다.
coverage와 seed stability가 더 좋은 rank `2`가 proxy keep이 됐다.

## 실행 조건

Command:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-proxy-review-fill
```

입력:

```bash
outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_template.json
```

출력:

```bash
outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.json
outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_summary.json
outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.md
```

출력은 generated artifact로 commit하지 않는다.

## Proxy Review Result

| candidate | selected by dead-air | score | timing | phrase | vocabulary | decision |
|---|:---:|---:|---|---|---|---|
| `margin_recovered_rank_1_seed_23_sample_1` | true | `0.251` | stiff | weak | thin | needs_followup |
| `margin_recovered_rank_2_seed_31_sample_5` | false | `0.698` | acceptable | strong | acceptable | keep |
| `margin_recovered_rank_3_seed_17_sample_3` | false | `0.564` | acceptable | strong | acceptable | needs_followup |

## 판단 근거

rank `1`:

- dead-air `0.375`로 가장 낮음
- note count `9`
- phrase coverage `0.437`
- onset coverage `0.312`
- sustained coverage `0.438`
- seed에 dead-air outlier 존재
- proxy decision: `needs_followup`

rank `2`:

- dead-air `0.444`
- note count `19`
- phrase coverage `0.937`
- onset coverage `0.500`
- sustained coverage `0.719`
- postprocess removal `0.095`
- seed failure 없음
- proxy decision: `keep`

rank `3`:

- phrase/onset/sustained coverage는 가장 높음
- dead-air `0.500`
- seed 내부에 dead-air failure와 unique pitch failure가 남아 있음
- proxy decision: `needs_followup`

## 해석

증명한 것:

- dead-air만으로 selected best를 고르면 phrase richness가 낮은 후보를 선택할 수 있다.
- 6-file 5-sample run에서 proxy 기준 가장 나은 후보는 rank `2` seed `31` sample `5`다.
- rank `2`는 seed 내부 failure가 없고, note count와 coverage가 가장 균형적이다.

리스크:

- 이 결과는 MIDI metric proxy review이며 실제 청감 review가 아니다.
- rank `2`를 broad model quality 증거로 주장할 수 없다.
- proxy keep은 다음 focused/context review로 넘길 수 있다는 뜻이다.

## 다음 작업

권장 이슈:

- `Stage B margin-recovered proxy keep consolidation`

목표:

- proxy keep 후보의 의미를 README/portfolio claim 관점에서 정리
- "생성 모델 완성"이 아니라 "raw generation + validation + candidate selection pipeline" claim으로 제한
- 다음 실험 경계와 resume bullet 근거를 분리
