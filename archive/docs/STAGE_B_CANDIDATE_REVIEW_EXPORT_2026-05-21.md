# Stage B Candidate Review Export

작성일: 2026-05-21

## Issue

- Issue: #47
- Branch: `issue-47-stage-b-review-package`
- Goal: Issue #45에서 생긴 top `coverage_chord` candidates를 manual listening/piano-roll review용 package로 export한다.

## Why

Issue #45는 first reviewable Stage B candidates를 만들었다.

하지만 다음 단계는 broad training이 아니다.

지금 필요한 판단은 metric이 아니라 실제 청취와 piano-roll review다.

따라서 ranking report에서 top candidates를 추출해, 사람이 바로 열어볼 수 있는 review manifest와 MIDI copy를 만든다.

## Implementation

Added:

- `scripts/export_stage_b_review_candidates.py`
- `tests/test_stage_b_review_export.py`

The script reads:

```text
outputs/stage_b_candidate_ranking/harness_stage_b_chord_aware_probe/candidate_rank_report.json
```

It writes:

```text
outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_manifest.json
outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_candidates.md
outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/midi/*.mid
```

Generated output files are local artifacts and are not committed.

## Command

```bash
./.venv/bin/python scripts/export_stage_b_review_candidates.py \
  --ranking_report outputs/stage_b_candidate_ranking/harness_stage_b_chord_aware_probe/candidate_rank_report.json \
  --run_id harness_stage_b_chord_aware_probe \
  --top_n 6 \
  --mode coverage_chord \
  --copy_midi
```

## Export Result

Selected candidates:

| review | source rank | mode | groups/bar | sample | score | notes | pitches | chord | bar chord | repeat | midi |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | coverage_chord | 4 | 2 | 96.6964 | 8 | 6 | 0.750 | 0.875 | 0.250 | `rank_01_coverage_chord_g4_s2.mid` |
| 2 | 2 | coverage_chord | 6 | 1 | 95.3410 | 12 | 8 | 0.667 | 0.917 | 0.333 | `rank_02_coverage_chord_g6_s1.mid` |
| 3 | 3 | coverage_chord | 8 | 1 | 94.1554 | 16 | 7 | 0.688 | 0.938 | 0.562 | `rank_03_coverage_chord_g8_s1.mid` |
| 4 | 4 | coverage_chord | 6 | 2 | 92.7577 | 12 | 6 | 0.667 | 0.917 | 0.500 | `rank_04_coverage_chord_g6_s2.mid` |
| 5 | 5 | coverage_chord | 8 | 3 | 92.5208 | 16 | 7 | 0.625 | 0.938 | 0.562 | `rank_05_coverage_chord_g8_s3.mid` |
| 6 | 6 | coverage_chord | 8 | 2 | 92.0929 | 16 | 7 | 0.625 | 0.938 | 0.562 | `rank_06_coverage_chord_g8_s2.mid` |

## Review Checklist

Listen and inspect for:

- solo-line shape
- phrase contour
- over-mechanical rhythm
- excessive high-register bias
- chord-tone correctness sounding too constrained
- one-note/two-note/chord-block/long-sustain failure

## Decision Boundary

If the top candidates sound like usable solo-line sketches, the next issue can design the generic jazz base training probe.

If they still sound mechanical, do not start broad training. Work on rhythm/motif-level behavior or evaluate a pretrained symbolic MIDI base first.

## Validation

Commands run:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_review_export
./.venv/bin/python -m compileall scripts/export_stage_b_review_candidates.py tests/test_stage_b_review_export.py
./.venv/bin/python scripts/export_stage_b_review_candidates.py --ranking_report outputs/stage_b_candidate_ranking/harness_stage_b_chord_aware_probe/candidate_rank_report.json --run_id harness_stage_b_chord_aware_probe --top_n 6 --mode coverage_chord --copy_midi
bash scripts/agent_harness.sh quick
```
