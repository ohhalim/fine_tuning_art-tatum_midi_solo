# Stage B Overlap-Free Review Export

작성일: 2026-05-22

## 목적

Issue #97은 generated review MIDI 후보가 solo-line이 아니라 chord block처럼 보이는 문제를 줄이기 위해, review export 단계에서 overlap-free solo-line variant를 생성한 단계다.

이 작업은 모델 품질 향상이 아니다. 원본 generated sample은 보존하고, 사람이 듣는 review MIDI만 겹침 없는 solo-line으로 정리해서 objective review가 제대로 비교할 수 있게 한다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `--overlap_free_review_midi`
  - `overlap_free_solo_notes`
  - `write_overlap_free_solo_midi`
  - review manifest `review_variant`
  - review manifest `review_postprocess_report`
- `scripts/build_listening_review_notes.py`
  - `review_metadata.review_variant`
  - `review_metadata.review_postprocess_report`
- `scripts/agent_harness.sh`
  - `stage-b-overlap-free-review-export`

## 동작 방식

원본 MIDI:

- `midi_path`에 보존한다.

청취 리뷰 MIDI:

- `review_midi_path`에 `*_overlap_free.mid`를 기록한다.
- 같은 onset의 중복 음은 대표 note만 남긴다.
- 다음 onset을 침범하는 duration은 다음 onset 직전까지 자른다.
- export report에 before/after max simultaneous notes와 trimmed note count를 남긴다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-overlap-free-review-export
```

이 하네스는 다음 순서로 실행된다.

1. overlap-free review MIDI export
2. objective MIDI note review
3. objective-aware listening review notes 생성
4. objective-aware aggregate 생성

## 결과

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_overlap_free_review_export/review_manifest.json`
- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_overlap_free_review_export/objective_midi_note_review.md`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_overlap_free_review_export/listening_review_aggregate.md`

요약:

- candidate count: `15`
- objective reviewable: `15`
- objective bucket counts:
  - clean: `5`
  - warning: `10`
- objective flag counts:
  - chromatic walk: `7`
  - duration pattern collapse: `6`
  - too stepwise/scalar: `4`
  - overlap/polyphonic: `0`

비교:

- 이전 objective flags review flow의 overlap/polyphonic count: `9`
- 이번 overlap-free review export의 overlap/polyphonic count: `0`

## 해석

overlap-free export는 명백한 piano-roll artifact를 제거했다.

하지만 이것이 좋은 jazz solo를 의미하지는 않는다.

남은 문제:

- `duration_pattern_collapse`
- `chromatic_walk`
- `too_stepwise_or_scalar`
- subjective phrase quality 미확정

따라서 다음 작업은 broad training이 아니라 duration/rhythm variation 또는 phrase/cadence vocabulary 개선이어야 한다.

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare tests.test_listening_review_notes
bash scripts/agent_harness.sh stage-b-overlap-free-review-export
bash scripts/agent_harness.sh quick
```
