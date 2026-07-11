# Stage B Data Motif Review Export

작성일: 2026-05-21

## 목적

Issue #65는 `hand_written_swing`과 `data_motif` baseline을 숫자로 비교했다.

이번 작업은 두 baseline의 MIDI 후보를 실제 piano-roll/listening review용으로 분리한다.

숫자상으로는 `data_motif`가 bar-pattern variation과 duration repetition에서 좋아졌지만, syncopation은 낮아졌다. 따라서 다음 판단은 report 수치만으로 하지 않고, mode가 명확히 드러나는 MIDI 파일을 직접 들어보는 것이다.

## 구현

변경 파일:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`
- `scripts/agent_harness.sh`

추가 하네스:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-review-export
```

생성 산출물:

- `outputs/stage_b_data_motif_compare/<run_id>/data_motif_compare_report.json`
- `outputs/stage_b_data_motif_compare/<run_id>/data_motif_compare_report.md`
- `outputs/stage_b_data_motif_review/<run_id>/review_manifest.json`
- `outputs/stage_b_data_motif_review/<run_id>/review_candidates.md`
- `outputs/stage_b_data_motif_review/<run_id>/named_midi/*.mid`

Generated artifacts는 커밋하지 않는다.

## Local Result

실행:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-review-export
```

결과:

- compare gate: `true`
- `hand_written_swing`: strict `3/3`
- `data_motif`: strict `3/3`
- review candidates: `6`
- named MIDI files:
  - `01_data_motif_rank_01_sample_01.mid`
  - `01_data_motif_rank_02_sample_02.mid`
  - `01_data_motif_rank_03_sample_03.mid`
  - `02_hand_written_swing_rank_01_sample_01.mid`
  - `02_hand_written_swing_rank_02_sample_02.mid`
  - `02_hand_written_swing_rank_03_sample_03.mid`

## Review 기준

이제 들어봐야 할 것은 "strict gate 통과"가 아니다.

확인할 질문:

- `data_motif`가 hand-written보다 phrase movement가 자연스러운가?
- syncopation 하락이 groove 약화로 들리는가?
- duration repetition이 줄어든 것이 실제로 덜 초급스럽게 들리는가?
- 여전히 코드톤/스케일 나열처럼 들리는가?
- 8-bar 안에서 call, continuation, landing이 느껴지는가?

## 다음 판단

가능한 분기:

- `data_motif`가 더 낫게 들리면, model constrained generation 쪽에 motif template sampling을 연결한다.
- 둘 다 초급 scale exercise처럼 들리면, contour보다 cadence/phrase-ending extraction을 먼저 강화한다.
- syncopation 하락이 크게 들리면, data-derived rhythm sampling에 syncopation floor를 추가한다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-data-motif-review-export
bash scripts/agent_harness.sh quick
```
