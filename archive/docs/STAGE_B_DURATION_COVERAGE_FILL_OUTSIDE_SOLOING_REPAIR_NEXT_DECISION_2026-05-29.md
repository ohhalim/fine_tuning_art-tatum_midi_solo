# Stage B Duration Coverage Fill Outside-Soloing Repair Next Decision

Issue #371은 outside-soloing repair objective evidence support를 다음 자동 작업 경계로 변환한 작업이다.

## Context

- Issue #369 boundary: `outside_soloing_repair_objective_evidence_support`
- objective support source candidates: `2/2`
- human/audio preference claimed: `false`
- broad model quality claimed: `false`
- listening preference status: pending

## Change

- outside-soloing repair next decision script 추가
- objective evidence support와 pending listening preference 경계 분리
- broader repeatability sweep 자동 진행 여부 기록
- 다음 issue selection constraints 기록
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| input boundary | `outside_soloing_repair_objective_evidence_support` |
| next boundary | `outside_soloing_repair_broader_repeatability_sweep` |
| auto progress allowed | `true` |
| critical user input required | `false` |
| human/audio preference claimed | `false` |
| broad model quality claimed | `false` |

## Objective Evidence

| item | value |
|---|---:|
| source candidates | `2` |
| qualified source candidates | `2` |
| dead-air preserved source candidates | `2` |
| chord-tone pass source candidates | `2` |
| non-chord run pass source candidates | `2` |
| interval pass source candidates | `2` |
| selected min chord-tone ratio | `1.000` |
| selected max non-chord run | `0` |
| selected max interval | `7` |

## Judgment

- selected repaired source `2/2` objective support 확보
- 청취 preference는 pending 상태 유지
- broader repeatability sweep으로 objective evidence 확장
- human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_outside_soloing_repair_next_decision.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-next-decision
```

## Output

- script: `scripts/decide_stage_b_duration_coverage_outside_soloing_repair_next_step.py`
- test: `tests/test_stage_b_duration_coverage_outside_soloing_repair_next_decision.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/harness_stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/harness_stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair broader repeatability sweep`
