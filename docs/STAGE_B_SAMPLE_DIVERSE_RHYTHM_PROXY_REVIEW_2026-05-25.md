# Stage B Sample-Diverse Rhythm Variation MIDI Proxy Review

작성일: 2026-05-25

## 목적

Issue #124는 Issue #122에서 sample diversity를 고친 rhythm variation 후보를 다시 같은 listening review notes schema로 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, duplicate note-sequence fields를 기준으로 한 proxy review다.
- `keep` 후보를 만들기보다, 다음 generation rule 병목을 정하는 단계다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` sample-diverse 후보 3개

사용한 생성 artifact:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy/sample_diverse_rhythm_variation_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy/listening_review_aggregate.md`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 0 |
| `needs_followup` | 6 |
| `reject` | 0 |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | objective-clean but still drops into C1-A1 register artifact |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest contour baseline, still rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | one unresolved large leap and low-register descent |
| `data_motif_rhythm_phrase_variation_rank_1_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | best sample-diverse variation candidate |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | independent but high IOI repetition |
| `data_motif_rhythm_phrase_variation_rank_3_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | independent but upper-register path is weak |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 14 | fragments/mechanical phrase quality remain |
| `fix_timing_grid` | 12 | every candidate is still timing-stiff |
| `increase_motif_variation` | 6 | repetition remains even without exact duplicates |

## Sample-Diverse Variation Notes

The duplicate problem is fixed:

- review candidates: `6`
- unique note sequences: `6`
- duplicate note sequences: `0`

Variation candidates:

| candidate | notes | pitches | sync | dur-var | ioi-var | ioi-rep | objective flags |
|---|---:|---:|---:|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_2` | 62 | 29 | 0.667 | 0.111 | 0.097 | 0.500 | `{}` |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | 62 | 22 | 0.730 | 0.111 | 0.113 | 0.565 | `{}` |
| `data_motif_rhythm_phrase_variation_rank_3_sample_1` | 60 | 21 | 0.694 | 0.097 | 0.115 | 0.426 | `{}` |

MIDI-note observations:

- rank 1 is the best representative: independent sequence, 62 notes, 29 unique pitches, no large leaps, no unresolved large leaps.
- rank 2 is independent and clean, but IOI repetition is high at `0.565`.
- rank 3 is independent and clean, but climbs to the upper register and stays there across later bars.

## 해석

Issue #122 repaired the exact-duplicate failure, and Issue #124 confirms the repaired candidates are independent review evidence.

However, this still does not create a musical `keep` candidate.

Remaining blockers:

- all 6 candidates are still `timing=too_stiff`
- 5 of 6 candidates are `fragment`
- variation candidates still have repeated IOI/template behavior
- phrase shape remains mechanical even when objective flags are clean

## Decision

Issue #124 conclusion:

- sample diversity is fixed enough to continue.
- no candidate should be promoted to `keep`.
- the next generation rule should target timing-grid repetition and phrase-template mechanicalness, not duplicate sequence repair.

Recommended next issue:

```text
Stage B rhythm variation timing-grid repetition repair
```

Target:

- reduce most-common IOI ratio while keeping objective-clean gate
- avoid long deterministic rest/onset template cells
- keep duplicate note sequence count at `0`
- preserve final guide landing and max interval bound

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy/sample_diverse_rhythm_variation_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_diverse_codex_proxy/sample_diverse_rhythm_variation_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
