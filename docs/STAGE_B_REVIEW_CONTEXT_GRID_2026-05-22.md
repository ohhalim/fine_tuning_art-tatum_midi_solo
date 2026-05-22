# Stage B Review Context and Straight Grid

작성일: 2026-05-22

## 배경

수동 리뷰에서 중요한 문제가 나왔다.

- solo-line만 들으면 chord progression이 들리지 않는다.
- 그래서 generated line이 in인지 out인지, chord tone인지 아닌지 판단하기 어렵다.
- swing/motif 후보는 DAW piano-roll에서 박자가 딱 맞지 않는 것처럼 들릴 수 있다.
- 리뷰용 MIDI는 음악적으로 멋진 후보만이 아니라, 판단 가능한 context를 제공해야 한다.

따라서 이번 작업은 모델 품질 개선이 아니라 review package 보정이다.

## 구현

변경 파일:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`
- `scripts/agent_harness.sh`

추가된 것:

- `straight_grid` baseline mode
- `chord_guide.mid`
- candidate별 `*_with_context.mid`
- review markdown의 solo/context MIDI 경로

하네스:

```bash
bash scripts/agent_harness.sh stage-b-review-context-grid
```

## 출력 구조

```text
outputs/stage_b_data_motif_review/harness_stage_b_review_context_grid/
  chord_guide.mid
  review_manifest.json
  review_candidates.md
  named_midi/
    01_data_motif_rank_01_sample_01.mid
    02_hand_written_swing_rank_01_sample_01.mid
    03_straight_grid_rank_01_sample_01.mid
  context_midi/
    01_data_motif_rank_01_sample_01_with_context.mid
    02_hand_written_swing_rank_01_sample_01_with_context.mid
    03_straight_grid_rank_01_sample_01_with_context.mid
```

`named_midi/`는 solo-only 후보다.

`context_midi/`는 다음 track을 같이 넣는다.

- chord guide
- bass root guide
- generated solo line

## Straight Grid의 의미

`straight_grid`는 "더 좋은 솔로" 후보가 아니다.

이 모드는 리뷰 기준점이다.

- 16th-position token grid 위에 딱 맞는 straight subdivision을 쓴다.
- swing/motif 후보가 박자상 이상하게 들리는지 비교하기 위한 reference다.
- strict gate가 낮게 나올 수 있다. 현재 dead-air metric은 start-to-start IOI를 기준으로 보기 때문에 8th-like straight line을 과하게 불리하게 본다.

따라서 `straight_grid`는 musical quality gate 통과 여부보다 timing reference로 본다.

## Local Result

실행:

```bash
bash scripts/agent_harness.sh stage-b-review-context-grid
```

결과:

- `data_motif`: strict `3/3`
- `hand_written_swing`: strict `3/3`
- `straight_grid`: exported as timing reference
- review candidates: `9`
- chord guide MIDI generated
- context MIDI generated for every candidate

## 다음 리뷰 방법

이제 solo-only MIDI만 듣지 않는다.

우선순위:

1. `chord_guide.mid`를 먼저 들어 chord progression 감각을 확인한다.
2. `context_midi/*data_motif*_with_context.mid`를 듣는다.
3. `context_midi/*hand_written_swing*_with_context.mid`와 비교한다.
4. `context_midi/*straight_grid*_with_context.mid`로 timing 기준을 확인한다.

판단할 질문:

- chord/bass guide 위에서 line이 in으로 들리는가?
- out처럼 들리는 음이 의도적인 tension/approach인지, 그냥 틀린 음인지 구별되는가?
- swing/motif 후보가 groove를 만들기보다 박자 밀림처럼 들리는가?
- straight-grid 후보는 너무 딱딱하지만 timing reference로는 더 편한가?
- 다음 생성기는 swing보다 straight quantized output을 먼저 유지해야 하는가?

## 다음 작업

리뷰 결과에 따라 분기한다.

- context 위에서도 line이 초급스럽다면 cadence/phrase-ending extraction을 먼저 한다.
- timing이 문제라면 generated output은 straight grid를 기본으로 두고 swing은 나중에 humanization 단계로 미룬다.
- chord context 위에서 out/in 구분이 어렵다면 bar별 chord-tone/tension annotation MIDI 또는 CSV를 추가한다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-review-context-grid
bash scripts/agent_harness.sh quick
```
