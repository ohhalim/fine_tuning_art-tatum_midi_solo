# Stage B Duration Variation Review

작성일: 2026-05-22

## 목적

Issue #99는 overlap-free review export 이후에도 남아 있던 `duration_pattern_collapse` 문제를 줄이기 위한 단계다.

이 작업은 좋은 jazz solo를 만든다는 선언이 아니다. 사람이 들어볼 review MIDI가 같은 길이의 음가만 반복하는지 먼저 제거하고, 남은 문제가 rhythm인지 pitch/phrase vocabulary인지 분리하기 위한 probe다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `varied_grid_position_duration_steps`
  - `varied_grid_tokens`
  - `varied_guide_tones_tokens`
  - `varied_grid` baseline mode
  - `varied_guide_tones` baseline mode
- `scripts/agent_harness.sh`
  - `stage-b-duration-variation-review`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - varied-duration step, token, guide-tone tests

## 동작 방식

기존 overlap-free export는 유지한다.

새 baseline은 다음 원칙을 따른다.

- onset은 16th grid 안에 둔다.
- duration은 단일 패턴으로 고정하지 않는다.
- 다음 onset을 침범하지 않도록 duration을 제한한다.
- `varied_guide_tones`는 guide-tone/cadence pitch vocabulary 위에 varied duration을 얹는다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-duration-variation-review
```

이 하네스는 다음 순서로 실행된다.

1. varied-duration baseline 포함 review MIDI export
2. objective MIDI note review
3. objective-aware listening review notes 생성
4. objective-aware aggregate 생성

## 결과

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_duration_variation_review/review_manifest.json`
- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_duration_variation_review/objective_midi_note_review.md`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_duration_variation_review/listening_review_aggregate.md`

요약:

- candidate count: `15`
- objective reviewable: `15`
- objective bucket counts:
  - clean: `8`
  - warning: `7`
- objective flag counts:
  - chromatic walk: `7`
  - too stepwise/scalar: `6`
  - duration pattern collapse: `0`
  - overlap/polyphonic: `0`

비교:

- Issue #97 overlap-free review export의 duration pattern collapse count: `6`
- Issue #99 duration variation review의 duration pattern collapse count: `0`
- Issue #97 clean count: `5`
- Issue #99 clean count: `8`

## 해석

duration collapse는 objective flag 기준으로 제거됐다.

하지만 이것이 jazz vocabulary 문제를 해결했다는 뜻은 아니다. 남아 있는 주요 문제는 다음이다.

- `chromatic_walk`
- `too_stepwise_or_scalar`
- 초급 scale exercise처럼 들릴 수 있는 pitch contour
- cadence target과 phrase vocabulary 부족

따라서 다음 작업은 duration을 더 만지는 것이 아니라, scalar/chromatic exercise 느낌을 줄이는 pitch contour / cadence / motif vocabulary 쪽이어야 한다.

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-duration-variation-review
bash scripts/agent_harness.sh quick
```
