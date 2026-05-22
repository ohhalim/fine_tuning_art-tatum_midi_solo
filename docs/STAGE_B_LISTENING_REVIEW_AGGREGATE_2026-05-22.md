# Stage B Listening Review Aggregate

작성일: 2026-05-22

## 배경

Issue #87에서 generated 후보를 사람이 들었을 때 기록할 `review_notes_template.json` schema를 만들었다.

이번 단계는 그 notes가 채워졌을 때 다음 generation rule 수정 방향을 숫자로 집계하는 도구를 추가한다.

중요한 경계:

- Codex가 subjective listening result를 임의로 채우지 않는다.
- pending-only notes에서는 generation rule을 바꾸라고 결론 내리지 않는다.
- 사람이 채운 `issues`와 `decision`을 다음 실험 분기 기준으로만 사용한다.

## 구현

새 스크립트:

```bash
python scripts/summarize_listening_review_notes.py
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-listening-review-aggregate
```

집계 항목:

- `decision_counts`
- `phrase_quality_counts`
- `timing_counts`
- `chord_fit_counts`
- `issue_counts`
- `source_metric_by_decision`
- `recommended_followups`

추천 follow-up code:

- `collect_listening_reviews`
- `fix_timing_grid`
- `increase_tension_approach_vocabulary`
- `improve_phrase_vocabulary`
- `tighten_chord_fit`
- `increase_motif_variation`
- `increase_density_or_coverage`

## 결과

Harness는 현재 pending-only template을 입력으로 검증한다.

| metric | value |
|---|---:|
| candidate count | 6 |
| reviewed | 0 |
| pending | 6 |
| has reviewed candidates | false |

Output:

```text
outputs/stage_b_listening_review_aggregate/harness_stage_b_listening_review_aggregate/listening_review_aggregate.json
outputs/stage_b_listening_review_aggregate/harness_stage_b_listening_review_aggregate/listening_review_aggregate.md
```

현재 follow-up:

```json
[
  {
    "code": "collect_listening_reviews",
    "reason": "No reviewed candidates are present, so generation rules should not be changed from this artifact alone.",
    "count": 0
  }
]
```

## 판단

이 단계의 의미는 "후보가 좋다"가 아니다.

의미:

- 사람이 채운 listening notes가 들어오면 다음 수정 방향을 자동 집계할 수 있다.
- `too_safe`, `too_scalar`, `too_mechanical`, `bad_timing`, `bad_chord_fit`를 분리할 수 있다.
- 아직 reviewed candidate가 없을 때는 rule change를 막는다.

## 다음 작업

실제 청취 결과가 채워진 notes가 생기면 aggregate 결과로 후속 issue를 분기한다.

예:

- `too_safe`가 많으면 tension/approach vocabulary를 늘린다.
- `too_mechanical` 또는 `too_scalar`가 많으면 phrase vocabulary와 motif variation을 고친다.
- `bad_timing`이 많으면 swing looseness보다 straight-grid timing을 우선한다.
- `bad_chord_fit`이 많으면 chord-aware pitch selection을 다시 좁힌다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_listening_review_notes tests.test_listening_review_aggregate
bash scripts/agent_harness.sh stage-b-listening-review-aggregate
```
