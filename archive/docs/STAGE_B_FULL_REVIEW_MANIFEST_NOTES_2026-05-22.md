# Stage B Full Review Manifest Listening Notes

작성일: 2026-05-22

## 배경

Issue #87의 listening review notes는 generated chord eval report를 기준으로 만들어져서 chord eval이 붙은 상위 6개 후보만 포함했다.

하지만 실제 청취 리뷰에서는 다음 reference 후보도 같이 비교해야 한다.

- `hand_written_swing`
- `straight_grid`
- `straight_guide_tones`
- `data_motif`
- `data_motif_guide_tones`

이번 단계는 `review_manifest.json` 전체를 notes template으로 변환해서, 사람이 들어야 할 파일명과 context MIDI path가 notes에 직접 남도록 한다.

## 구현

확장된 스크립트:

```bash
python scripts/build_listening_review_notes.py --review_manifest ...
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-full-review-notes
```

기존 경로는 유지된다.

```bash
python scripts/build_listening_review_notes.py --generated_chord_eval_report ...
```

## Notes Candidate Fields

`review_manifest` 기반 notes candidate는 다음 필드를 가진다.

- `candidate_id`
  - 예: `data_motif_rank_1_sample_1`
- `review_metadata`
  - `mode`
  - `review_rank`
  - `sample_index`
  - `sample_seed`
  - `valid`
  - `strict_valid`
- `review_files`
  - `midi_path`
  - `source_midi_path`
  - `context_midi_path`
- `source_metrics`
  - note/pitch count
  - dead-air ratio
  - syncopation ratio
  - duration/IOI diversity
  - tension/root ratios
- `listening`
  - 사람이 채울 pending review fields

## 결과

Harness result:

| metric | value |
|---|---:|
| candidate count | 15 |
| reviewed | 0 |
| pending | 15 |

Output:

```text
outputs/stage_b_listening_review_notes/harness_stage_b_full_review_notes/review_notes_template.json
```

첫 candidate:

```text
data_motif_rank_1_sample_1
```

첫 context MIDI:

```text
outputs/stage_b_data_motif_review/harness_stage_b_full_review_notes/context_midi/01_data_motif_rank_01_sample_01_with_context.mid
```

마지막 candidate:

```text
straight_guide_tones_rank_3_sample_3
```

## 판단

이 단계의 의미:

- 이제 review notes가 실제 review package 전체와 1:1로 대응된다.
- 사람이 "어떤 파일을 들어야 하는지"를 notes만 보고 알 수 있다.
- hand-written swing과 straight-grid reference도 subjective review 대상에 포함된다.
- Codex는 여전히 subjective listening result를 임의 작성하지 않는다.

## 다음 작업

다음 generation rule 변경은 사람이 채운 full review notes를 aggregate한 뒤 결정한다.

우선순위 예:

- `bad_timing`이 많으면 swing looseness를 줄이고 grid landing을 강화한다.
- `too_scalar`가 많으면 scale-walk vocabulary를 제한하고 phrase contour/motif를 강화한다.
- `too_safe`가 많으면 guide-tone만이 아니라 tension/approach resolution을 늘린다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_listening_review_notes tests.test_listening_review_aggregate
bash scripts/agent_harness.sh stage-b-full-review-notes
bash scripts/agent_harness.sh quick
```
