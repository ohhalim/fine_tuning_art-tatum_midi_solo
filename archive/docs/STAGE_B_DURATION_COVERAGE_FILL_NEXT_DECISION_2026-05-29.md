# Stage B Duration Coverage Fill Next Decision

Issue #349는 user listening review consolidation 이후 다음 작업 경계를 정리한 작업이다.

## Context

- Issue #347 boundary: `midi_evidence_and_single_user_listening_support_duration_coverage_fill_keep`
- preferred candidate: `duration_coverage_fill_keep`
- MIDI/user preference aligned: `true`
- rendered audio file count: `2`
- single user review: `true`
- broad model quality claimed: `false`

## Change

- next decision summary script 추가
- repeatability vs repair decision rule 정의
- single-candidate support와 broad-quality not-proven boundary 유지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| preferred candidate | `duration_coverage_fill_keep` |
| next boundary | `broader_repeatability_sweep` |
| auto progress allowed | `true` |
| critical user input required | `false` |
| broad model quality claimed | `false` |

## Decision

- fill candidate는 MIDI evidence와 single-user listening review에서 같은 방향으로 지지됨
- 아직 multi-seed repeatability가 미검증
- 다음 경계: broader repeatability sweep

## Not Proven

- multi-seed repeatability
- multi-reviewer preference
- audio rendered quality
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_next_decision.py
bash scripts/agent_harness.sh stage-b-duration-coverage-next-decision
```

## Output

- script: `scripts/decide_stage_b_duration_coverage_next_step.py`
- test: `tests/test_stage_b_duration_coverage_next_decision.py`
- summary: `outputs/stage_b_duration_coverage_fill_next_decision/harness_stage_b_duration_coverage_fill_next_decision/stage_b_duration_coverage_fill_next_decision.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill broader repeatability sweep`
