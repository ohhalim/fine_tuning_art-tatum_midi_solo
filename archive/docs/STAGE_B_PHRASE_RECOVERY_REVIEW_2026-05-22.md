# Stage B Phrase Recovery Review

작성일: 2026-05-22

## 목적

Issue #105는 Issue #103에서 드러난 `unresolved_large_leaps` 문제를 줄이기 위해 `phrase_recovery` baseline을 추가한 단계다.

이 작업은 큰 도약을 금지하지 않는다. 큰 도약 뒤에 반대 방향의 작은 회복 움직임을 넣어, 도약 나열 exercise가 아니라 phrase motion에 가까워지는지 objective metric으로 확인한다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `phrase_recovery` baseline mode
  - `phrase_recovery_pitch_classes`
  - `recovery_pitch_after_large_leap`
  - `phrase_recovery_tokens`
- `scripts/agent_harness.sh`
  - `stage-b-phrase-recovery-review`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - `phrase_recovery` mode parsing
  - unresolved large leap ratio regression test

## 동작 방식

`phrase_recovery`는 `phrase_cadence`와 같은 varied-duration grid와 guide/tension/cadence pitch-class vocabulary를 사용한다.

추가 규칙:

- 이전 interval이 `7` semitone 이상이면 large leap로 본다.
- 다음 note는 가능한 경우 반대 방향으로 움직인다.
- recovery interval은 `1-5` semitone 범위를 선호한다.
- recovery pitch는 current/next chord의 guide tone, non-root chord tone, tension tone 후보 안에서 고른다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-phrase-recovery-review
```

이 하네스는 `phrase_cadence`와 `phrase_recovery`를 같은 조건에서 비교한다.

## 결과

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_phrase_recovery_review/review_manifest.json`
- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_phrase_recovery_review/objective_midi_note_review.md`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_recovery_review/listening_review_aggregate.md`

요약:

- candidate count: `6`
- objective bucket counts:
  - clean: `3`
  - warning: `3`
- objective flag counts:
  - unresolved large leaps: `3`
- mode flag counts:
  - `phrase_cadence`: unresolved large leaps `3`
  - `phrase_recovery`: no objective flags

비교:

- `phrase_cadence` unresolved large leap ratio:
  - `0.750-0.757`
- `phrase_recovery` unresolved large leap ratio:
  - `0.000-0.048`

## 해석

`phrase_recovery`는 Issue #103에서 드러난 large-leap recovery 문제를 objective 기준으로 해결한다.

하지만 이것은 아직 subjective jazz quality를 의미하지 않는다. 다음 확인 지점은 다음이다.

- recovery motion이 실제로 자연스럽게 들리는지
- guide-tone exercise처럼 건조하지 않은지
- data-derived motif rhythm과 결합했을 때도 clean 상태를 유지하는지

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-phrase-recovery-review
bash scripts/agent_harness.sh quick
```
