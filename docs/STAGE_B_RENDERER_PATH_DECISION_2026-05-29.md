# Stage B Renderer Path Decision

Issue #341은 renderer unavailable 상태에서 local audio render attempt 전 필요한 decision boundary를 정리한 작업이다.

## Context

- Issue #339 tooling status: `renderer_unavailable`
- fluidsynth available: `false`
- timidity available: `false`
- soundfont exists: `false`
- system modified: `false`
- package install executed: `false`
- download executed: `false`
- audio render attempted: `false`

## Change

- renderer path decision summary script 추가
- ready/missing/unavailable 상태별 next boundary 정의
- user/system dependency required flag 기록
- install/download/render attempt guard 유지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| tooling status | `renderer_unavailable` |
| decision | `renderer_path_or_install_approval_required` |
| critical user input required | `true` |
| blocked reason | `renderer_unavailable` |
| package install executed | `false` |
| external download executed | `false` |
| audio render attempted | `false` |

## Allowed Paths

- existing `fluidsynth` path + `.sf2/.sf3` soundfont path 제공
- existing `timidity` path 제공
- audio render skip 후 MIDI evidence only 경로 유지

## Not Proven

- audio rendered quality
- human/audio preference
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_renderer_path_decision.py
bash scripts/agent_harness.sh stage-b-renderer-path-decision
```

## Output

- script: `scripts/decide_stage_b_renderer_path.py`
- test: `tests/test_stage_b_renderer_path_decision.py`
- summary: `outputs/stage_b_renderer_path_decision/harness_stage_b_renderer_path_decision/stage_b_renderer_path_decision.json`

## Next

- renderer/soundfont path 제공 또는 설치 승인 후 `Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render attempt`
- audio render skip 결정 시 MIDI evidence only 경로 유지
