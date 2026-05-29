# Stage B Duration Coverage Fill Local Audio Render Attempt

Issue #343은 source/fill MIDI를 FluidSynth와 GeneralUser GS soundfont로 WAV 렌더한 작업이다.

## Context

- Issue #341 decision: `renderer_path_or_install_approval_required`
- user approval: FluidSynth install and soundfont download 허용
- renderer: `/opt/homebrew/bin/fluidsynth`
- soundfont: `~/.local/share/soundfonts/generaluser-gs/v1.471.sf2`
- soundfont source: `https://github.com/ropensci/fluidsynth/releases/download/generaluser-gs-v1.471/generaluser-gs-v1.471.zip`
- soundfont license reference: `https://github.com/mrbumpy409/GeneralUser-GS/blob/main/documentation/LICENSE.txt`
- render target: duration/coverage fill source vs keep MIDI package

## Change

- local audio render attempt script 추가
- source/fill WAV 생성
- WAV sample rate, channel count, duration, size, sha256 검증
- rendered audio file path summary 기록
- audio quality/human preference claim guard 유지

## Result

| item | value |
|---|---:|
| render attempted | `true` |
| rendered audio file count | `2` |
| technical WAV validation | `true` |
| sample rate | `44100` |
| source duration seconds | `6.474` |
| fill duration seconds | `6.474` |
| audio rendered quality claimed | `false` |
| human/audio preference claimed | `false` |

## Rendered Files

- source: `outputs/stage_b_duration_coverage_fill_local_audio_render_attempt/harness_stage_b_duration_coverage_fill_local_audio_render_attempt/audio/source_constrained_partial.wav`
- fill: `outputs/stage_b_duration_coverage_fill_local_audio_render_attempt/harness_stage_b_duration_coverage_fill_local_audio_render_attempt/audio/duration_coverage_fill_keep.wav`

## Not Proven

- audio rendered quality
- human/audio preference
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_local_audio_render_attempt.py
bash scripts/agent_harness.sh stage-b-local-audio-render-attempt
```

## Output

- script: `scripts/render_stage_b_duration_coverage_fill_audio.py`
- test: `tests/test_stage_b_local_audio_render_attempt.py`
- summary: `outputs/stage_b_duration_coverage_fill_local_audio_render_attempt/harness_stage_b_duration_coverage_fill_local_audio_render_attempt/stage_b_duration_coverage_fill_local_audio_render_attempt.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill user listening review fill`
- user review input 필요: source/fill preference, timing, phrase, vocabulary, notes
