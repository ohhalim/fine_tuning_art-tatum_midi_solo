# Stage B Margin-Recovered Candidate Review Export

작성일: 2026-05-28

## 결론

Issue #238의 6-file / 5-sample repeatability 결과에서 seed별 best candidate 3개를 objective review table로 추출했다.
MIDI 파일은 commit하지 않고, summary JSON에서 metric만 읽어 review export를 생성했다.

| rank | selected | seed | sample | seed strict | outliers | dead-air | notes | pitches | phrase | onset | sustained | removal |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | true | `23` | `1` | `4/5` | `1` | `0.375` | `9` | `4` | `0.437` | `0.312` | `0.438` | `0.357` |
| `2` | false | `31` | `5` | `5/5` | `0` | `0.444` | `19` | `4` | `0.937` | `0.500` | `0.719` | `0.095` |
| `3` | false | `17` | `3` | `3/5` | `1` | `0.500` | `17` | `4` | `1.000` | `0.594` | `0.844` | `0.227` |

selected best는 seed `23`, sample `1`이다.

다만 objective metric만 보면 rank `2` seed `31` sample `5`는 dead-air가 낮고 note count, phrase/onset/sustained coverage가 더 높다.
현재 rank는 dead-air를 가장 먼저 보는 기준이므로, 다음 단계에서는 rank `1`과 rank `2`를 listening review 대상으로 같이 봐야 한다.

## 실행 조건

Command:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-review-export
```

입력:

```bash
outputs/stage_b_raw_generation_repeatability/issue_238_stage_b_candidate_count_margin_recovery/repeatability_summary.json
```

출력:

```bash
outputs/stage_b_margin_recovered_review_export/harness_stage_b_margin_recovered_review_export/candidate_review_export.json
outputs/stage_b_margin_recovered_review_export/harness_stage_b_margin_recovered_review_export/candidate_review_export.md
```

위 출력은 generated artifact로 commit하지 않는다.

## Source Summary

| 항목 | 값 |
|---|---:|
| source run | `issue_238_stage_b_candidate_count_margin_recovery` |
| repeatability gate | true |
| total strict samples | `12/15` |
| strict pass-rate | `0.800` |
| dead-air outlier rate | `0.133` |
| strict margin warning seeds | 없음 |
| exported candidates | `3` |
| selected best rank | `1` |

## Seed Failure Reasons

| seed | failure |
|---:|---|
| `17` | dead-air `1`, unique pitch `1` |
| `23` | dead-air `1` |
| `31` | none |

## 해석

증명한 것:

- 5-sample margin-recovered run에서 seed별 best 후보를 reviewable metric table로 분리했다.
- selected best는 dead-air 기준으로 seed `23`, sample `1`이다.
- seed `31`, sample `5`는 selected best보다 dead-air는 높지만, note count와 coverage가 더 높아 listening 비교 가치가 있다.
- seed `17`, sample `3`은 strict-valid지만 seed 내부 failure가 남아 있어 안정성은 가장 낮다.

리스크:

- review export는 objective metric 정리일 뿐, 청감 품질을 보장하지 않는다.
- MIDI 파일은 generated output이므로 commit하지 않는다.
- 현재 rank 기준은 dead-air 우선이며, phrase richness나 musicality를 직접 반영하지 않는다.

## 다음 작업

권장 이슈:

- `Stage B margin-recovered candidate listening review notes`

목표:

- rank `1`, `2`, `3` 후보를 listening review note template로 정리
- dead-air 우선 rank와 실제 청감 선호가 일치하는지 확인
