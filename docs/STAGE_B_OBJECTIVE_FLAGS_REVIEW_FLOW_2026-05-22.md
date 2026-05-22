# Stage B Objective Flags Review Flow

작성일: 2026-05-22

## 목적

Issue #95는 objective MIDI note review 결과를 listening review notes와 aggregate report에 연결한 단계다.

이 작업은 MIDI가 "재즈답다"를 자동 판정하지 않는다. 목적은 사람이 듣기 전에 명백히 문제가 있는 후보를 `problem`, 상대적으로 들을 가치가 있는 후보를 `warning`으로 분리해 review priority를 만드는 것이다.

## 구현

변경 사항:

- `scripts/review_midi_note_objectives.py`
  - `objective_penalty`
  - `objective_priority_score`
  - `objective_reviewable`
  - `objective_bucket`
- `scripts/build_listening_review_notes.py`
  - `--objective_midi_review_report`
  - 후보별 `objective_review` 첨부
- `scripts/summarize_listening_review_notes.py`
  - objective flag counts
  - objective bucket counts
  - objective review priority table
- `scripts/agent_harness.sh`
  - `stage-b-objective-flags-review-flow`

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-objective-flags-review-flow
```

이 하네스는 다음 순서로 실행된다.

1. data-guide hybrid review package 생성
2. objective MIDI note review 생성
3. objective-aware listening review notes 생성
4. objective-aware aggregate 생성

## 결과

출력:

- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_objective_flags_review_flow/objective_midi_note_review.json`
  - `outputs/stage_b_objective_midi_review/harness_stage_b_objective_flags_review_flow/objective_midi_note_review.md`
- review notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_objective_flags_review_flow/review_notes_template.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_objective_flags_review_flow/listening_review_aggregate.json`
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_objective_flags_review_flow/listening_review_aggregate.md`

요약:

- candidate count: `15`
- objective reviewable count: `6`
- objective buckets:
  - problem: `9`
  - warning: `6`
- objective flags:
  - chromatic walk: `7`
  - duration pattern collapse: `9`
  - overlap/polyphonic: `9`
  - too stepwise/scalar: `4`

## 해석

현재 후보는 이전보다 valid MIDI에 가까워졌지만, 좋은 jazz solo라고 볼 수는 없다.

특히:

- `overlap_polyphonic` 후보는 solo-line review 우선순위에서 낮춘다.
- `duration_pattern_collapse` 후보는 grid는 맞아도 rhythm이 너무 단조롭다.
- `chromatic_walk` / `too_stepwise_or_scalar` 후보는 scale exercise처럼 들릴 가능성이 높다.
- `warning` 후보도 최종 성공이 아니라 "문제는 있지만 들어볼 수 있는 후보"다.

## 다음 판단

다음 generation 작업은 broad training이 아니라 다음 두 갈래 중 하나여야 한다.

1. solo-line export에서 overlap/polyphonic이 생기지 않도록 고친다.
2. straight-grid 후보의 duration collapse를 줄이는 rhythm/duration variation을 추가한다.

subjective listening review는 이 objective priority를 참고해서 채우되, Codex가 임의로 "좋다/나쁘다"를 확정하지 않는다.

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_objective_midi_note_review tests.test_listening_review_notes tests.test_listening_review_aggregate
bash scripts/agent_harness.sh stage-b-objective-flags-review-flow
bash scripts/agent_harness.sh quick
```
