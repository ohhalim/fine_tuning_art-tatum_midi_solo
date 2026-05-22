# Stage B Data-Motif Guide-Tone Hybrid

작성일: 2026-05-22

## 배경

Issue #71에서 `straight_guide_tones`를 추가해 timing과 harmonic vocabulary를 분리했다.

수동 리뷰 기준은 다음으로 좁혀졌다.

- swing/humanization은 MIDI 입력 후보에서 박자가 흐트러져 들린다.
- straight-grid는 박자는 명확하지만 너무 교과서적이다.
- data-derived motif는 rhythm variation이 낫지만 pitch가 아직 jazz vocabulary처럼 들린다고 보기 어렵다.

따라서 이번 단계는 `data_motif`의 rhythm template을 유지하면서 pitch만 guide-tone/cadence 문법으로 제한하는 hybrid 후보를 만든다.

## 추가된 Mode

```text
data_motif_guide_tones
```

구성:

- rhythm: real phrase window에서 추출한 data-derived motif rhythm template
- duration: data motif duration template
- register contour: data motif contour를 target pitch 방향 참고로만 사용
- pitch class: current chord guide tone, chord tone, tension, limited approach tone
- strong beat: 3도/7도 guide tone으로 제한
- weak beat: chord/tension/short approach만 허용

## 검증

실행:

```bash
bash scripts/agent_harness.sh stage-b-data-guide-hybrid
bash scripts/agent_harness.sh quick
```

확인하는 것:

- 기존 `data_motif` vs `hand_written_swing` compare gate가 깨지지 않는다.
- `data_motif_guide_tones`가 context MIDI review package로 export된다.
- strong beat pitch class가 current chord guide tone에 들어간다.
- non-chord pitch가 연속 chromatic run으로 이어지지 않는다.

결과:

- compare gate: passed
- `data_motif`: strict `3/3`
- `data_motif_guide_tones`: strict `3/3`
- `hand_written_swing`: strict `3/3`
- `straight_grid`: strict `0/3`, timing reference
- `straight_guide_tones`: strict `0/3`, timing/pitch reference
- `data_motif_guide_tones` note count: `63`
- `data_motif_guide_tones` unique pitch count: `23-24`
- `data_motif_guide_tones` chord-tone ratio: `0.797`
- `data_motif_guide_tones` tension ratio: `0.062`
- `data_motif_guide_tones` root-tone ratio: `0.000`
- `data_motif_guide_tones` unique bar-position pattern ratio: `1.000`

## Review Files

생성 위치:

```text
outputs/stage_b_data_motif_review/harness_stage_b_data_guide_hybrid/
```

중요 파일:

```text
review_candidates.md
chord_guide.mid
context_midi/02_data_motif_guide_tones_rank_01_sample_01_with_context.mid
context_midi/02_data_motif_guide_tones_rank_02_sample_02_with_context.mid
context_midi/02_data_motif_guide_tones_rank_03_sample_03_with_context.mid
named_midi/02_data_motif_guide_tones_rank_01_sample_01.mid
named_midi/02_data_motif_guide_tones_rank_02_sample_02.mid
named_midi/02_data_motif_guide_tones_rank_03_sample_03.mid
```

## 판단 기준

이번 후보에서 들어볼 것은 세 가지다.

1. `straight_guide_tones`보다 덜 교과서적인가?
2. `data_motif`보다 scale/chromatic exercise 느낌이 줄었는가?
3. chord/bass guide 위에서 strong beat이 안정적으로 들리는가?

좋으면 다음 단계는 이 hybrid를 기본 review candidate로 삼는다.

나쁘면 다음 단계는 코드가 아니라 데이터 쪽이다.

- real jazz phrase window에서 bar별 guide-tone landing 통계를 뽑는다.
- cadence cell을 hand-written이 아니라 reference-derived로 바꾼다.
- 또는 chord annotation/pitch-role review table을 먼저 추가한다.
