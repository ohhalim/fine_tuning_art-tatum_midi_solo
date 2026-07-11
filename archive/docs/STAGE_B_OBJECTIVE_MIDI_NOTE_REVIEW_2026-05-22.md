# Stage B Objective MIDI Note Review

작성일: 2026-05-22

## 배경

사람이 듣기 전에 Codex가 MIDI 노트 자체로 확인할 수 있는 문제가 있다.

이번 단계는 generated review MIDI를 직접 읽어 다음 항목을 객관 리포트로 만든다.

- note count / unique pitch count
- max active notes / polyphonic tick ratio
- 16th grid alignment
- duration pattern collapse
- stepwise / chromatic walk ratio
- chord-tone / tension / outside / root ratio
- first 16 note preview

중요한 경계:

- 이 리포트는 subjective listening review가 아니다.
- "재즈답다"를 판정하지 않는다.
- 다만 사람이 듣기 전에 걸러낼 수 있는 machine-observable 문제를 잡는다.

## 구현

새 스크립트:

```bash
python scripts/review_midi_note_objectives.py
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-objective-midi-review
```

출력:

```text
outputs/stage_b_objective_midi_review/harness_stage_b_objective_midi_review/objective_midi_note_review.json
outputs/stage_b_objective_midi_review/harness_stage_b_objective_midi_review/objective_midi_note_review.md
```

## 결과

Harness result:

| flag | count |
|---|---:|
| chromatic_walk | 7 |
| duration_pattern_collapse | 9 |
| overlap_polyphonic | 9 |
| too_stepwise_or_scalar | 4 |

Mode-level interpretation:

- `data_motif`
  - overlap/polyphonic이 있다.
  - 일부 후보는 chromatic walk 성향도 있다.
- `data_motif_guide_tones`
  - stepwise/chromatic 위험은 낮아졌다.
  - 하지만 overlap/polyphonic은 남아 있다.
- `hand_written_swing`
  - overlap/polyphonic이 가장 크다.
  - stepwise/scalar와 chromatic walk가 강하다.
  - duration pattern collapse도 있다.
- `straight_grid`
  - max active note는 1이라 solo-line 형태는 깨끗하다.
  - 하지만 chromatic/scale exercise 성향이 강하다.
  - duration pattern collapse가 있다.
- `straight_guide_tones`
  - chromatic/stepwise 위험은 줄었다.
  - 하지만 duration pattern collapse가 남아 있다.

## 판단

사용자 piano-roll review와 objective metrics가 맞는다.

구체적으로:

- "박자가 안 맞는다"로 느껴진 `hand_written_swing`은 실제로 16th grid 밖이 아니다.
- 문제는 off-grid보다 overlap/polyphonic, scalar/chromatic walk, repeated duration pattern 쪽에 가깝다.
- `straight_grid`는 grid는 맞지만 scale/chromatic exercise처럼 보인다는 지적이 metrics로 잡힌다.
- `straight_guide_tones`는 pitch vocabulary는 개선됐지만 rhythm이 너무 균일하다.

## 다음 작업

다음 generation rule 수정은 다음 순서가 맞다.

1. solo-line review 후보에서는 overlap/polyphonic을 제거하거나 gate에서 강하게 감점한다.
2. `straight_grid` 계열은 duration/onset variation 없이는 reference 이상으로 쓰지 않는다.
3. chromatic walk ratio와 stepwise ratio가 높은 후보는 listening review 전에 낮은 우선순위로 보낸다.
4. 실제 청취 notes는 objective flags와 함께 채운다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_objective_midi_note_review tests.test_listening_review_notes tests.test_listening_review_aggregate
bash scripts/agent_harness.sh stage-b-objective-midi-review
bash scripts/agent_harness.sh quick
```
