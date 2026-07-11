# Stage B Guide-Tone Cadence Candidate

작성일: 2026-05-22

## 배경

Issue #69 review package를 듣고 다음 문제가 확인됐다.

- `hand_written_swing`과 `data_motif` 후보는 "swing"이라고 부르기에는 MIDI grid가 흔들려 들린다.
- `straight_grid`는 박자는 맞지만 scale/chromatic exercise처럼 들린다.
- solo-only MIDI보다 chord/bass context MIDI가 리뷰에는 더 낫다.

따라서 이번 단계는 swing을 더 고치는 것이 아니라, straight quantized timing을 유지한 채 pitch vocabulary를 guide-tone/cadence 중심으로 제한하는 것이다.

## 구현

추가된 baseline mode:

```text
straight_guide_tones
```

핵심 규칙:

- position은 straight 8th grid를 유지한다.
- strong beat position `0`, `4`, `8`, `12`는 현재 chord의 guide tone으로 제한한다.
- guide tone은 주로 3도와 7도다.
- root는 가능한 피하고 non-root chord tone, tension, 짧은 approach tone을 사용한다.
- approach tone은 연속으로 나오지 않게 제한한다.
- bar 마지막 note는 다음 chord guide tone으로 가는 짧은 approach로 둔다.

## 검증

실행:

```bash
bash scripts/agent_harness.sh stage-b-guide-tone-cadence
bash scripts/agent_harness.sh quick
```

결과:

- compare gate: passed
- `hand_written_swing`: strict `3/3`
- `data_motif`: strict `3/3`
- `straight_grid`: exported as timing reference, strict `0/3`
- `straight_guide_tones`: exported as timing/pitch reference, strict `0/3`
- `straight_guide_tones` note count: `64`
- `straight_guide_tones` unique pitch count: `26-29`
- `straight_guide_tones` chord-tone ratio: `0.656`
- `straight_guide_tones` tension ratio: `0.172`
- `straight_guide_tones` root-tone ratio: `0.000`

`straight_grid`와 `straight_guide_tones`가 strict false인 주된 이유는 current metric의 dead-air 판단이 straight 8th grid reference에 맞지 않기 때문이다. 이 후보들은 "모델 성공 후보"가 아니라 timing/pitch review reference로 본다.

## Review Files

생성 위치:

```text
outputs/stage_b_data_motif_review/harness_stage_b_guide_tone_cadence/
```

중요 파일:

```text
review_candidates.md
chord_guide.mid
context_midi/04_straight_guide_tones_rank_01_sample_01_with_context.mid
context_midi/04_straight_guide_tones_rank_02_sample_02_with_context.mid
context_midi/04_straight_guide_tones_rank_03_sample_03_with_context.mid
```

## 판단

이번 결과는 jazz solo 완성이 아니다.

의미는 다음이다.

- swing/humanization을 억지로 넣기 전에 quantized MIDI review 후보를 분리했다.
- scale/chromatic 나열 문제를 pitch grammar로 줄이는 첫 기준선을 만들었다.
- 앞으로의 musical review는 `data_motif`와 `straight_guide_tones`를 chord context 위에서 비교해야 한다.

다음 단계:

- `straight_guide_tones`를 들어보고 너무 교과서적이면 motif rhythm은 data-derived로 유지하고 strong-beat pitch만 guide-tone cadence로 제한한다.
- strict gate는 generated swing 후보용과 straight reference용을 분리한다.
- chord별 in/out annotation이 아직 부족하면 review markdown에 bar/chord/pitch-role table을 추가한다.
