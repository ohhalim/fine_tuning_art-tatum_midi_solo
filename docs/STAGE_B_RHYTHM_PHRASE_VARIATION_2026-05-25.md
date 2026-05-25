# Stage B Rhythm/Phrase Vocabulary Variation

작성일: 2026-05-25

## 목적

Issue #118은 Issue #116 contour repair MIDI-note proxy review에서 드러난 문제를 좁혀서 검증한다.

Issue #116 결과:

- `keep`: `0`
- `too_stiff`: `6`
- `too_mechanical`: `6`
- `too_repetitive`: `6`
- `weak_phrase`: `5`

따라서 이번 probe는 landing repair를 반복하지 않고, 다음을 테스트한다.

- duration/IOI template rigidity 완화
- phrase vocabulary variation
- repaired phrase가 C1/A#1 solo register로 떨어지는 문제 방지
- objective-clean contour/landing 조건 유지

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- broad training 성공 근거가 아니다.
- Brad style adaptation 성공 근거가 아니다.
- MIDI objective와 review package를 만드는 engineering probe다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `data_motif_rhythm_phrase_variation` baseline mode 추가
  - data-derived rhythm template을 variable slot boundary로 재배치
  - duration steps에 deterministic variation을 더하되 next onset 전까지만 유지
  - penultimate note에서 guide landing으로 접근하는 approach pitch class 사용
  - solo register floor/ceiling: `48-84`
  - variation mode의 normal/cadence pitch interval bound를 `6` semitones로 제한
- `scripts/agent_harness.sh`
  - `stage-b-rhythm-phrase-variation`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - mode parsing
  - register floor
  - resolved final landing
  - rhythm IOI variation guard

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

비교 mode:

- `data_motif_contour_landing_repair`
- `data_motif_rhythm_phrase_variation`

출력:

- compare report:
  - `outputs/stage_b_data_motif_compare/harness_stage_b_rhythm_phrase_variation/data_motif_compare_report.md`
- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- review candidates:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.md`
- listening notes template:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation/review_notes_template.json`

## 결과

Compare summary:

| mode | samples | strict | landing | max interval | resets | sync | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data_motif_contour_landing_repair` | 3 | 3 | 1.000 | 7 | 0 | 0.625 | 0.062 | 0.375 | 0.079 | 0.429 | 0.443 | 0.016 |
| `data_motif_rhythm_phrase_variation` | 3 | 3 | 1.000 | 6 | 0 | 0.694 | 0.097 | 0.339 | 0.115 | 0.426 | 0.371 | 0.016 |

Objective MIDI review:

- candidate count: `6`
- flag counts: `{}`
- `data_motif_rhythm_phrase_variation` candidates:
  - note count: `60`
  - pitch range: `51-80`
  - repeated pitch interval ratio: `0.000`
  - unresolved large leap ratio: `0.000`
  - final landing: `3/3`
  - max interval: `6`
- listening review notes:
  - reviewed count: `0`
  - pending count: `6`

## 해석

`data_motif_rhythm_phrase_variation` improves the narrow objective rhythm target.

Improved compared with `data_motif_contour_landing_repair`:

- syncopated onset ratio: `0.625` -> `0.694`
- duration diversity ratio: `0.062` -> `0.097`
- most common duration ratio: `0.375` -> `0.339`
- IOI diversity ratio: `0.079` -> `0.115`
- max interval: `7` -> `6`
- pitch floor: previous repaired candidates reached C1/G1; variation candidates stay at `>=51`

Preserved:

- strict review gate: `3/3`
- final landing resolved: `3/3`
- objective MIDI flags: `{}`
- abrupt register resets: `0`
- repeated pitch interval ratio: `0.000`
- unresolved large leap ratio: `0.000`

Tradeoffs:

- note count is `60`, not `63`, because variable slot boundaries produce a slightly sparser line.
- average tension ratio drops from `0.443` to `0.371`.
- this still does not prove musical quality; listening review is pending.

## Decision

Issue #118 is a useful objective improvement, but not a musical success claim.

Recommended next issue:

```text
Stage B rhythm/phrase variation MIDI-note proxy review
```

Next review should compare the new variation candidates against the previous contour repair candidates and decide whether:

- rhythm variation is perceptible enough to reduce `too_stiff`
- the lower tension ratio makes the line too safe
- the register floor improves phrase plausibility
- the slightly lower note count feels sparse or acceptable

Do not move to broad training or product/backend work yet.

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py tests/test_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
bash scripts/agent_harness.sh quick
```
