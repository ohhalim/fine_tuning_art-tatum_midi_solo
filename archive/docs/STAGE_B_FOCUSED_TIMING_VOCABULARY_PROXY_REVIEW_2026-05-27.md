# Stage B Focused Timing Vocabulary Proxy Review

작성일: 2026-05-27

## 목적

Issue #186은 Issue #184 focused timing/vocabulary follow-up repair 후보를 MIDI-note/context 기준으로 다시 판단한 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, context MIDI path, objective MIDI metrics, Issue #184 short-cell tradeoff를 기준으로 판단했다.
- proxy `keep`은 final musical keep이 아니라 focused context review package로 넘길 수 있다는 뜻이다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_focused_timing_vocab_followup_repair/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_focused_timing_vocab_followup_repair/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_proxy_review/focused_timing_vocab_repaired_review_notes.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_focused_timing_vocab_proxy_review/listening_review_aggregate.json`

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 1 |
| `needs_followup` | 3 |
| `reject` | 2 |

Quality counts:

| field | result |
|---|---|
| phrase quality | `phrase=3`, `fragment=2`, `exercise=1` |
| timing | `acceptable=3`, `too_stiff=3` |
| chord fit | `fits=6` |
| objective bucket | `clean=6` |
| objective flags | `{}` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | contour baseline still reads as register-drifting fragment |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `exercise` | `too_stiff` | `fits` | `needs_followup` | clean landing, but still rigid contour-drill behavior |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `reject` | contour baseline problem and unresolved large-leap risk remain |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `acceptable` | `fits` | `needs_followup` | max interval stays safe, but adjacent repeats and bar-grid feel remain |
| `data_motif_rhythm_phrase_variation_rank_2_sample_2` | `phrase` | `acceptable` | `fits` | `needs_followup` | useful sync/tension, but adjacent repeated notes increased |
| `data_motif_rhythm_phrase_variation_rank_3_sample_3` | `phrase` | `acceptable` | `fits` | `keep` | best repaired proxy keep candidate for focused context review |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 12 | still the strongest remaining blocker |
| `fix_timing_grid` | 6 | contour baseline and repaired candidates still expose grid-derived timing |
| `increase_motif_variation` | 4 | repeated short-cell behavior remains in non-keep candidates |

## Proxy Keep Candidate

Selected candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- decision: `keep`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `20`
- source syncopated onset ratio: `0.703`
- source duration diversity ratio: `0.078`
- source most-common IOI ratio: `0.397`
- objective stepwise interval ratio: `0.460`
- objective tension ratio: `0.453`
- root-tone ratio: `0.031`
- final landing: `guide`
- max interval: `4`

Why keep only as proxy:

- It is the repaired candidate with the widest pitch vocabulary.
- It keeps objective-clean status, no large leaps, and final guide landing.
- It reduces the worst 4-note/8-note duplicated-cell concern from the focused follow-up repair.
- It still has mechanical timing/vocabulary risk, so it is not final musical evidence.

## Decision

Issue #186 conclusion:

- Promote exactly one repaired candidate to proxy `keep`.
- Do not claim final musical quality.
- The next issue should isolate the proxy keep candidate into a focused context review package.
- Broad training, Brad style adaptation, backend/API, and audio pivot remain premature.

Recommended next issue:

```text
Stage B focused timing vocabulary proxy keep focused package
```

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_proxy_review/focused_timing_vocab_repaired_review_notes.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_proxy_review/focused_timing_vocab_repaired_review_notes.json
```
