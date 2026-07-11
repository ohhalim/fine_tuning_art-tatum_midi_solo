# Stage B Focused Timing Vocabulary Follow-up Proxy Review

작성일: 2026-05-27

## Context

Issue #198은 Issue #196 focused listening follow-up repair 후보를 MIDI-note/context 기준으로 다시 판단한 proxy review다.

Issue #196 repair의 목적은 다음이었다.

- adjacent pitch repeat 제거
- duplicated 3-note/4-note pitch-class cell 감소
- final guide landing, max interval, objective-clean guardrail 유지
- source tension 하락과 short-cell tradeoff를 fresh proxy review에서 분리

## Scope

포함한 작업:

- Issue #196 review manifest 기반 proxy listening notes 생성
- 6개 후보 전체에 structured review decision 기록
- aggregate follow-up signal 생성
- proxy keep 후보가 focused context package로 갈 수 있는지 판단

범위 밖:

- 실제 오디오 청취 리뷰
- broad training
- Brad style adaptation claim
- backend/API/product MVP
- generated artifact commit

## Inputs

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_focused_timing_vocab_listening_followup_repair/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_focused_timing_vocab_listening_followup_repair/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_proxy_review/focused_timing_vocab_followup_repaired_review_notes.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_focused_timing_vocab_followup_proxy_review/listening_review_aggregate.json`

## Decision Summary

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
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | low-register contour baseline, not repair evidence |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `exercise` | `too_stiff` | `fits` | `needs_followup` | objective-clean but still contour-drill behavior |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `reject` | low-register drift and unresolved large-leap risk remain |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `acceptable` | `fits` | `needs_followup` | adjacent repeats removed, but 3/4-note cells increased |
| `data_motif_rhythm_phrase_variation_rank_2_sample_2` | `phrase` | `acceptable` | `fits` | `keep` | best repaired proxy keep for focused context review |
| `data_motif_rhythm_phrase_variation_rank_3_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | adjacent repeats removed, but source tension and 3-note cells regress |

## Selected Proxy Keep

Selected candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- decision: `keep`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `19`
- pitch range: `G3-G5`
- adjacent repeated pitch count: `0`
- duplicated 3-note pitch-class chunks: `0`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`
- source syncopated onset ratio: `0.719`
- source tension ratio: `0.344`
- objective tension ratio: `0.469`
- objective outside ratio: `0.016`
- final landing: `D5`
- max interval: `4`

Why keep only as proxy:

- It is the only repaired variation candidate where adjacent repeat, 3-note cell replay, 4-note cell replay, and 8-note cell replay are all `0`.
- It preserves the focused-context register range and final guide landing.
- It remains objective-clean with no review flags.
- It still has mechanical timing risk, so it is not final musical quality evidence.

## Risk

Known tradeoffs:

- `data_motif_rhythm_phrase_variation_rank_1_sample_1` increased duplicated 3-note and 4-note cells.
- `data_motif_rhythm_phrase_variation_rank_3_sample_3` increased duplicated 3-note cells and source tension fell to `0.281`.
- The aggregate still recommends `improve_phrase_vocabulary`, `fix_timing_grid`, and `increase_motif_variation`.
- Proxy keep is a package input only; it does not justify broad training or style adaptation.

## Follow-up

Next issue:

```text
Stage B focused timing vocabulary follow-up proxy keep focused package
```

Acceptance for the next boundary:

- isolate only `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- copy solo/context MIDI to a focused package
- preserve objective first-note summary and proxy decision metadata
- keep final quality claims conservative

## Validation

Executed validation:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_followup_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_proxy_review/focused_timing_vocab_followup_repaired_review_notes.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_followup_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_proxy_review/focused_timing_vocab_followup_repaired_review_notes.json
```
