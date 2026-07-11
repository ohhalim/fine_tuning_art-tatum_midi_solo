# Stage B Data Motif Phrase Recovery

작성일: 2026-05-22

## 목적

Issue #107은 Issue #105의 `phrase_recovery` pitch grammar를 data-derived motif rhythm template과 결합한 단계다.

목표는 hand-written grid가 아니라 실제 MIDI phrase window에서 추출한 rhythm shape 위에서도 unresolved large leap 문제가 줄어드는지 확인하는 것이다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `data_motif_phrase_recovery` baseline mode
  - `data_motif_phrase_recovery_tokens`
- `scripts/agent_harness.sh`
  - `stage-b-data-motif-phrase-recovery-review`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - mode parsing test
  - data rhythm preservation and leap recovery test

## 동작 방식

`data_motif_phrase_recovery`는 다음을 결합한다.

- rhythm:
  - extracted motif template의 position/duration
  - data-derived bar-position variation 유지
- pitch:
  - phrase/cadence pitch-class cells
  - large leap 뒤 반대 방향 small recovery
  - current/next chord의 guide tone, non-root chord tone, tension tone 후보

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-phrase-recovery-review
```

비교 mode:

- `data_motif_guide_tones`
- `data_motif_phrase_recovery`
- `phrase_recovery`

## 결과

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_data_motif_phrase_recovery_review/review_manifest.json`
- objective report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_data_motif_phrase_recovery_review/objective_midi_note_review.md`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_data_motif_phrase_recovery_review/listening_review_aggregate.md`

요약:

- candidate count: `9`
- objective bucket counts:
  - clean: `6`
  - warning: `3`
- objective flag counts:
  - unresolved large leaps: `3`
- mode flag counts:
  - `data_motif_guide_tones`: unresolved large leaps `3`
  - `data_motif_phrase_recovery`: no objective flags
  - `phrase_recovery`: no objective flags

비교:

- `data_motif_guide_tones` unresolved large leap ratio:
  - `0.583-0.652`
- `data_motif_phrase_recovery` unresolved large leap ratio:
  - `0.000-0.045`
- `data_motif_phrase_recovery` tension ratio:
  - `0.476-0.524`

## 해석

`data_motif_phrase_recovery`는 data-derived rhythm shape를 유지하면서 phrase recovery objective risk를 줄였다.

이제 다음 판단은 pure objective gate가 아니라 listening review에 가깝다. 특히 다음을 들어봐야 한다.

- recovery motion이 음악적으로 자연스러운지
- data-derived rhythm이 실제로 덜 기계적으로 느껴지는지
- tension ratio가 높아져도 random color tone처럼 들리지 않는지

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-data-motif-phrase-recovery-review
bash scripts/agent_harness.sh quick
```
