# Stage B Rhythm/Phrase Variation Sample Diversity Repair

작성일: 2026-05-25

## 목적

Issue #122는 Issue #120 MIDI-note proxy review에서 확인된 exact duplicate 후보 문제를 고친다.

Issue #120 발견:

- `data_motif_rhythm_phrase_variation` rank 1-3 후보가 MIDI note/start/duration sequence 기준으로 완전히 동일했다.
- rank 1은 representative follow-up candidate였지만, rank 2/3은 독립적인 review evidence가 아니어서 reject했다.

이번 목표:

- sample seed가 실제 note sequence에 반영되게 한다.
- review export에서 exact duplicate note sequence를 표시한다.
- objective-clean gate를 유지하면서 독립 후보 3개를 만든다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `data_motif_rhythm_phrase_variation_tokens()`에서 seed-derived `seed_variation`을 추가했다.
  - seed가 rhythm template row, contour template row, slot boundary, duration variation, pitch-cell selection, penultimate approach target에 반영된다.
  - `midi_note_sequence_signature()`를 추가해 MIDI note/start/end/pitch sequence를 hash로 기록한다.
  - `annotate_duplicate_note_sequences()`를 추가해 review manifest 후보에 duplicate 여부를 기록한다.
  - review manifest top-level에 `unique_note_sequence_count`, `duplicate_note_sequence_count`를 기록한다.
  - review candidates markdown에 `duplicate` column을 추가한다.
- `tests/test_stage_b_data_motif_generation_compare.py`
  - seed `17/18/19`가 서로 다른 rhythm variation note sequence를 만드는지 검증한다.
  - review export가 duplicate MIDI sequence를 표시하는지 검증한다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

출력:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- review candidates:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## 결과

Review export duplicate summary:

| metric | value |
|---|---:|
| candidate count | 6 |
| unique note sequences | 6 |
| duplicate note sequences | 0 |

Variation mode result:

| candidate | notes | pitches | sync | dur-var | dur-rep | ioi-var | ioi-rep | landing | max interval | duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_2` | 62 | 29 | 0.667 | 0.111 | 0.397 | 0.097 | 0.500 | guide | 6 | false |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | 62 | 22 | 0.730 | 0.111 | 0.476 | 0.113 | 0.565 | guide | 6 | false |
| `data_motif_rhythm_phrase_variation_rank_3_sample_1` | 60 | 21 | 0.694 | 0.097 | 0.339 | 0.115 | 0.426 | guide | 6 | false |

Compare summary:

| mode | samples | strict | landing | max interval | sync | dur-var | ioi-var | ioi-rep | objective flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `data_motif_contour_landing_repair` | 3 | 3 | 3/3 | 7 | 0.625 | 0.062 | 0.079 | 0.429 | `{}` |
| `data_motif_rhythm_phrase_variation` | 3 | 3 | 3/3 | 6 | 0.697 | 0.106 | 0.108 | 0.497 | `{}` |

## 해석

고친 점:

- exact duplicate 후보 문제가 사라졌다.
- ranked review package가 이제 독립 MIDI note sequence 6개를 표시한다.
- variation 후보 3개 모두 strict/objective-clean 상태를 유지한다.
- final landing, max interval, register bound는 유지된다.

남은 위험:

- `data_motif_rhythm_phrase_variation_rank_2_sample_3`의 IOI repetition이 `0.565`로 높다.
- variation mode 평균 IOI repetition도 `0.497`이라 timing stiffness risk는 아직 남아 있다.
- 이 수정은 sample diversity repair이지 musical quality proof가 아니다.

## Decision

Issue #122는 sample-level duplicate problem을 해결했다.

다음 단계는 새 independent candidates를 다시 proxy review하는 것이다.

Recommended next issue:

```text
Stage B sample-diverse rhythm variation MIDI-note proxy review
```

다음 review에서 확인할 것:

- independent variation 후보가 이전 exact duplicate 후보보다 review evidence로 유효한지
- rank 2의 high IOI repetition이 실제로 too-stiff로 들릴지
- duration/IOI variation이 phrase quality 개선으로 이어졌는지
- still no `keep`이면 timing-grid repair 또는 motif cadence vocabulary를 별도 issue로 분리할지

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py tests/test_stage_b_data_motif_generation_compare.py
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Commit 전 필수 quick harness도 실행한다.
