# Stage B Phrase/Cadence Review

작성일: 2026-05-22

## 목적

Issue #101은 Issue #99 이후 남은 `chromatic_walk`와 `too_stepwise_or_scalar` 문제를 줄이기 위한 단계다.

이 작업은 review set을 더 좋은 jazz solo라고 주장하는 단계가 아니다. duration collapse와 overlap artifact를 줄인 다음, MIDI가 scale/chromatic exercise처럼 들리는 문제를 objective flag 기준으로 줄일 수 있는지 확인하는 probe다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `phrase_cadence` baseline mode
  - `phrase_cadence_pitch_class_cells`
  - `nearest_phrase_pitch_for_pitch_class`
  - selected-mode strict gate
- `scripts/agent_harness.sh`
  - `stage-b-phrase-cadence-review`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - `phrase_cadence` mode parsing
  - selected-mode gate test
  - scalar/chromatic interval ratio test

## 동작 방식

`phrase_cadence`는 Issue #99의 varied-duration grid를 유지한다.

pitch는 다음 원칙을 따른다.

- strong target은 guide tone과 non-root chord tone을 우선한다.
- color/tension tone을 포함해 너무 안전한 chord-tone 나열을 피한다.
- register target을 번갈아 사용해 stepwise scale run을 줄인다.
- 이전 2개 pitch 반복을 피하고, 가능하면 3-10 semitone 범위의 phrase interval을 선호한다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
```

이 하네스는 다음 순서로 실행된다.

1. `phrase_cadence`, `varied_guide_tones`, `data_motif`, `data_motif_guide_tones` review MIDI export
2. overlap-free review MIDI 생성
3. objective MIDI note review
4. objective-aware listening review notes 생성
5. objective-aware aggregate 생성

## 결과

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_phrase_cadence_review/review_manifest.json`
- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_phrase_cadence_review/objective_midi_note_review.md`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_cadence_review/listening_review_aggregate.md`

요약:

- candidate count: `12`
- objective reviewable: `12`
- objective bucket counts:
  - clean: `11`
  - warning: `1`
- objective flag counts:
  - chromatic walk: `1`
  - too stepwise/scalar: `0`
  - duration pattern collapse: `0`
  - overlap/polyphonic: `0`

비교:

- Issue #99 duration variation review:
  - candidate count: `15`
  - clean: `8`
  - warning: `7`
  - chromatic walk: `7`
  - too stepwise/scalar: `6`
- Issue #101 phrase/cadence review:
  - candidate count: `12`
  - clean: `11`
  - warning: `1`
  - chromatic walk: `1`
  - too stepwise/scalar: `0`

## 해석

objective MIDI flag 기준으로 scalar/chromatic exercise 문제는 크게 줄었다.

하지만 이 결과는 아직 subjective jazz quality를 증명하지 않는다. 특히 `phrase_cadence`는 register leap을 의도적으로 넣었기 때문에 실제 청취에서는 다음을 확인해야 한다.

- leap이 자연스러운 phrase로 들리는지
- chord context 위에서 cadence가 분명한지
- 너무 건조한 guide-tone exercise처럼 들리지 않는지
- data-derived motif rhythm과 충분히 결합할 가치가 있는지

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
bash scripts/agent_harness.sh quick
```
