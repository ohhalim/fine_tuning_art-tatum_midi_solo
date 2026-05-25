# Stage B Data-Derived Contour/Cadence Landing Repair

작성일: 2026-05-25

## 목적

이 probe는 clean MIDI-note proxy review에서 드러난 두 문제를 좁혀서 검증한다.

- phrase contour가 큰 register jump 뒤에 끊기는 문제
- final landing이 current chord guide/chord tone으로 해결되지 않는 문제

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- Brad style adaptation 성공이 아니다.
- rhythm stiffness와 repeated duration/rest template 문제를 해결한 단계도 아니다.
- 목적은 `data_motif_phrase_recovery` baseline 위에서 contour/landing repair가 objective MIDI review를 악화시키지 않는지 보는 것이다.

## 구현

변경 사항:

- `scripts/run_stage_b_data_motif_generation_compare.py`
  - `data_motif_contour_landing_repair` baseline mode
  - data-derived rhythm template과 contour template 결합
  - bar 마지막 note를 current/next chord guide tone 또는 non-root chord tone으로 landing
  - 같은 음 반복을 피하면서 가까운 pitch class를 고르는 bounded pitch selector
  - `final_landing_resolved`, `final_landing_role`, `max_abs_interval`, `abrupt_register_reset_count` metrics
- `scripts/agent_harness.sh`
  - `stage-b-contour-landing-repair`
- `tests/test_stage_b_data_motif_generation_compare.py`
  - mode parsing
  - landing/contour/register smoothing
  - repeated pitch ratio guard

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-contour-landing-repair
```

비교 mode:

- `data_motif_phrase_recovery`
- `data_motif_contour_landing_repair`

출력:

- compare report:
  - `outputs/stage_b_data_motif_compare/harness_stage_b_contour_landing_repair/data_motif_compare_report.md`
- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_contour_landing_repair/review_manifest.json`
- review candidates:
  - `outputs/stage_b_data_motif_review/harness_stage_b_contour_landing_repair/review_candidates.md`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_contour_landing_repair/objective_midi_note_review.md`
- listening notes template:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair/review_notes_template.json`

## 결과

Compare summary:

| mode | samples | strict | final landing | max interval | resets | sync | bar-var | dur-var | ioi-var | tension | root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `data_motif_contour_landing_repair` | 3 | 3 | 1.000 | 7 | 0 | 0.625 | 1.000 | 0.062 | 0.079 | 0.443 | 0.016 |
| `data_motif_phrase_recovery` | 3 | 3 | 0.333 | 13 | 0 | 0.625 | 1.000 | 0.062 | 0.079 | 0.438 | 0.005 |

Objective MIDI review:

- candidate count: `6`
- flag counts: `{}`
- `data_motif_contour_landing_repair` top candidates:
  - final landing role: `guide`
  - max interval: `7`
  - abrupt register resets: `0`
  - unique pitches: `25-26`
- `data_motif_phrase_recovery` comparison candidates:
  - final landing roles: `guide`, `tension`, `tension`
  - max interval: `12-13`
  - unique pitches: `19-23`

Listening review notes:

- reviewed count: `0`
- pending count: `6`
- recommended follow-up: `collect_listening_reviews`

## 해석

`data_motif_contour_landing_repair`는 이번 probe의 좁은 objective target을 만족한다.

- final landing unresolved 문제는 `1/3`에서 `3/3`으로 개선됐다.
- max interval은 `13`에서 `7`로 줄었다.
- abrupt register reset은 `0`으로 유지됐다.
- repeated-pitch objective flag는 발생하지 않았다.
- overlap-free review MIDI 기준 objective flag count는 `{}`다.

하지만 아직 musical success는 아니다.

- rhythm metrics는 baseline과 같다.
- duration diversity ratio는 `0.062`로 낮다.
- IOI diversity ratio는 `0.079`로 낮다.
- listening review는 아직 모두 pending이다.
- 따라서 이 결과만으로 broad training이나 product scope로 넘어가면 안 된다.

## 다음 판단

다음 단계는 generation rule을 또 바꾸기 전에 context MIDI를 기준으로 review notes를 채우는 것이다.

확인할 질문:

- landing repair가 실제로 phrase ending처럼 들리는가?
- max interval 감소가 contour continuity로 체감되는가?
- 여전히 grid-stiff/exercise-like이면 rhythm template diversity나 rest/duration variation을 다음 issue로 분리해야 하는가?

## 검증

실행한 검증:

```bash
./.venv/bin/python -m py_compile scripts/run_stage_b_data_motif_generation_compare.py tests/test_stage_b_data_motif_generation_compare.py
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-contour-landing-repair
bash scripts/agent_harness.sh quick
```
