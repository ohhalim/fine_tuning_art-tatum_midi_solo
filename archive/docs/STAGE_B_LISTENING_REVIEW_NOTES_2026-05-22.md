# Stage B Listening Review Notes Schema

작성일: 2026-05-22

## 배경

Issue #85에서 review markdown과 chord eval summary를 한 파일로 결합했다.

이번 단계는 실제 청취 리뷰 결과를 후보별로 구조화해서 기록할 schema와 template generator를 만든다.

중요한 경계:

- Codex가 청취 판단을 임의로 작성하지 않는다.
- 이 schema는 사람이 들은 결과를 일관되게 기록하기 위한 양식이다.
- raw MIDI/generated output은 계속 `outputs/` artifact로만 둔다.

## 구현

새 스크립트:

```bash
python scripts/build_listening_review_notes.py
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-listening-review-notes
```

실행 순서:

1. combined review markdown with chord eval을 생성한다.
2. generated chord eval report에서 candidate list를 읽는다.
3. `review_notes_template.json`을 만든다.
4. enum/status validation을 수행한다.

출력:

```text
outputs/stage_b_listening_review_notes/harness_stage_b_listening_review_notes/review_notes_template.json
outputs/stage_b_listening_review_notes/harness_stage_b_listening_review_notes/review_notes_summary.json
outputs/stage_b_listening_review_notes/harness_stage_b_listening_review_notes/review_notes_summary.md
```

## Schema

각 candidate는 다음 정보를 가진다.

Source metrics:

- `note_count`
- `unique_pitch_count`
- `chord_tone_ratio`
- `tension_ratio`
- `approach_ratio`
- `outside_ratio`

Listening fields:

- `status`
  - `pending`
  - `reviewed`
- `phrase_quality`
  - `pending`
  - `phrase`
  - `fragment`
  - `exercise`
  - `invalid`
- `timing`
  - `pending`
  - `acceptable`
  - `too_stiff`
  - `too_loose`
  - `off_grid`
- `chord_fit`
  - `pending`
  - `fits`
  - `too_safe`
  - `too_outside`
  - `unclear`
- `issues`
  - `too_safe`
  - `too_scalar`
  - `too_mechanical`
  - `weak_phrase`
  - `bad_timing`
  - `bad_chord_fit`
  - `too_repetitive`
  - `too_sparse`
  - `other`
- `decision`
  - `pending`
  - `keep`
  - `reject`
  - `needs_followup`
- `notes`

## 결과

Harness result:

| metric | value |
|---|---:|
| candidate count | 6 |
| reviewed | 0 |
| pending | 6 |
| keep | 0 |
| needs_followup | 0 |
| reject | 0 |

첫 candidate template 예:

```json
{
  "candidate_id": "data_motif_rank_1_sample_1",
  "source_metrics": {
    "note_count": 32,
    "unique_pitch_count": 18,
    "chord_tone_ratio": 0.5,
    "tension_ratio": 0.21875,
    "approach_ratio": 0.28125,
    "outside_ratio": 0.0
  },
  "listening": {
    "status": "pending",
    "phrase_quality": "pending",
    "timing": "pending",
    "chord_fit": "pending",
    "issues": [],
    "decision": "pending",
    "notes": ""
  }
}
```

## 판단

이제 다음 청취 리뷰는 자유서술이 아니라 structured note로 남길 수 있다.

의미:

- "좋다/별로다"가 아니라 왜 별로인지 분리할 수 있다.
- timing 문제인지, chord fit 문제인지, phrase vocabulary 문제인지 구분할 수 있다.
- 다음 generation rule 수정의 근거가 생긴다.

## 다음 작업

다음은 사람이 실제로 combined review markdown을 보면서 `review_notes_template.json`을 채우는 단계다.

그 후 자동화할 수 있는 작업:

```text
filled listening review notes를 aggregate해서 다음 generation rule을 결정한다.
```

예:

- `too_safe`가 많으면 tension/approach vocabulary를 늘린다.
- `too_mechanical`이 많으면 motif/rhythm variation을 강화한다.
- `bad_timing`이 많으면 straight-grid 기본값을 유지한다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_listening_review_notes
bash scripts/agent_harness.sh stage-b-listening-review-notes
bash scripts/agent_harness.sh quick
```
