# Stage B Duration Coverage Fill Outside-Soloing Repair Objective Evidence Consolidation

Issue #369는 outside-soloing repair 후보의 objective evidence를 하나의 claim boundary로 정리한 작업이다.

## Context

- Issue #363 boundary: `outside_soloing_pitch_role_repair_candidates`
- Issue #367 boundary: `outside_soloing_repair_audio_review_pending`
- review input present: `false`
- objective auto progress allowed: `true`
- required boundary: objective evidence only, no human/audio preference claim

## Change

- outside-soloing repair objective evidence consolidation script 추가
- selected repaired 후보 `2`개 objective gate 집계
- dead-air preservation, chord-tone ratio, non-chord run, max interval 경계 분리
- human/audio preference와 broad model quality claim guard 유지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `outside_soloing_repair_objective_evidence_support` |
| source candidates | `2` |
| qualified source candidates | `2` |
| dead-air preserved source candidates | `2` |
| chord-tone pass source candidates | `2` |
| non-chord run pass source candidates | `2` |
| interval pass source candidates | `2` |
| selected min chord-tone ratio | `1.000` |
| selected max non-chord run | `0` |
| selected max interval | `7` |
| human/audio preference claimed | `false` |
| broad model quality claimed | `false` |

## Selected Sources

| sample seed | policy | dead-air | chord-tone | non-chord run | max interval | interval delta |
|---:|---|---:|---:|---:|---:|---:|
| `155` | `contour_resolution` | `0.3333` | `1.000` | `0` | `7` | `+1` |
| `131` | `contour_resolution` | `0.3529` | `1.000` | `0` | `5` | `-6` |

## Judgment

- pitch-role objective evidence는 selected repaired source `2/2`에서 support
- dead-air preserved source candidates: `2/2`
- chord-tone pass source candidates: `2/2`
- non-chord run pass source candidates: `2/2`
- interval pass source candidates: `2/2`
- human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-objective-evidence
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.py`
- test: `tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair next decision`
