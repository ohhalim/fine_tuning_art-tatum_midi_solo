# Stage B Margin-Recovered Listening Review Notes

작성일: 2026-05-28

## 결론

Issue #240의 margin-recovered review export를 기반으로 listening review notes template을 생성했다.
후보 3개는 모두 `pending` 상태이며, 실제 청감 판단은 아직 기록하지 않았다.

| 항목 | 결과 |
|---|---:|
| candidate count | `3` |
| selected best count | `1` |
| reviewed count | `0` |
| pending count | `3` |
| pending decisions | `3` |

## 실행 조건

Command:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-listening-notes
```

입력:

```bash
outputs/stage_b_margin_recovered_review_export/harness_stage_b_margin_recovered_review_export/candidate_review_export.json
```

출력:

```bash
outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_template.json
outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_summary.json
outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_template.md
```

출력은 generated artifact로 commit하지 않는다.

## Candidate Notes

| candidate | selected | seed | sample | rank | dead-air | notes | pitches | phrase | onset | sustained | decision |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `margin_recovered_rank_1_seed_23_sample_1` | true | `23` | `1` | `1` | `0.375` | `9` | `4` | `0.437` | `0.312` | `0.438` | pending |
| `margin_recovered_rank_2_seed_31_sample_5` | false | `31` | `5` | `2` | `0.444` | `19` | `4` | `0.937` | `0.500` | `0.719` | pending |
| `margin_recovered_rank_3_seed_17_sample_3` | false | `17` | `3` | `3` | `0.500` | `17` | `4` | `1.000` | `0.594` | `0.844` | pending |

## 해석

증명한 것:

- margin-recovered 후보 3개를 동일한 listening review schema로 묶었다.
- selected best 후보는 하나만 존재하도록 검증했다.
- objective metric과 MIDI path는 review note 안에 보존했다.
- 실제 청감 review field는 pending으로 유지했다.

리스크:

- 이 단계는 review 준비이며, 청감 품질 판정이 아니다.
- MIDI 파일 자체는 output artifact이며 commit하지 않는다.
- rank `1`이 실제 listening preference에서도 best라고 확정한 것은 아니다.

## 다음 작업

권장 이슈:

- `Stage B margin-recovered MIDI proxy review fill`

목표:

- rank `1`, `2`, `3` 후보를 MIDI note/context evidence 기준으로 proxy review
- timing, phrase, jazz vocabulary, decision field를 채움
- rank 기준과 proxy review 판단이 일치하는지 기록
