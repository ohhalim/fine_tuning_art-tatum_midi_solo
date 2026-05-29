# Stage B Local Audio Render Tooling Setup

Issue #339는 local audio render attempt 전 renderer/soundfont readiness를 점검한 작업이다.

## Context

- Issue #337 local audio render package 완료
- current local render status: `renderer_unavailable`
- planned audio outputs: `2`
- render attempted: `false`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`

## Change

- local audio render tooling readiness script 추가
- renderer/soundfont probe summary 추가
- system modification, package install, download, audio render attempt를 모두 `false`로 검증
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| tooling status | current local probe `renderer_unavailable` |
| fluidsynth available | `false` |
| timidity available | `false` |
| soundfont exists | `false` |
| system modified | `false` |
| package install executed | `false` |
| download executed | `false` |
| audio render attempted | `false` |

## Boundary

- renderer/soundfont 준비 전 audio render attempt 금지
- package manager install 자동 실행 제외
- generated audio artifact commit 제외
- audio rendered quality와 human/audio preference 미검증 유지

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_local_audio_render_tooling.py
bash scripts/agent_harness.sh stage-b-local-audio-render-tooling
```

## Output

- script: `scripts/check_stage_b_local_audio_render_tooling.py`
- test: `tests/test_stage_b_local_audio_render_tooling.py`
- summary: `outputs/stage_b_local_audio_render_tooling/harness_stage_b_local_audio_render_tooling/stage_b_local_audio_render_tooling.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill renderer path decision`
- renderer/soundfont 준비 후 `Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render attempt`
