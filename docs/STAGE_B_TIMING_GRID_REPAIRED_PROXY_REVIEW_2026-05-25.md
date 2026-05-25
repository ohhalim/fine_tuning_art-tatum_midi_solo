# Stage B Timing-Grid Repaired Rhythm MIDI Proxy Review

작성일: 2026-05-25

## 목적

Issue #128은 Issue #126 timing-grid repetition repair 후보를 같은 listening review notes schema로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, duplicate note-sequence fields를 기준으로 한 proxy review다.
- `keep` 후보를 만들기보다 다음 generation rule 병목을 정하는 단계다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` timing-grid repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_timing_grid_repaired_codex_proxy/timing_grid_repaired_rhythm_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_timing_grid_repaired_codex_proxy/listening_review_aggregate.md`

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
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register artifact remains |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest baseline, still rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | unresolved leap and low-register descent |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | objective-clean but IOI repetition still high |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | best sign from timing repair, still grid-quantized |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | lower IOI repetition, but conservative diversity |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 14 | phrase/mechanical problems still dominate |
| `fix_timing_grid` | 12 | all candidates are still too stiff |
| `increase_motif_variation` | 6 | repetitive template behavior remains |

## Timing Repair Findings

Timing repair helped the objective MIDI surface:

- duplicate note sequences: `0`
- objective flags: `{}`
- variation max interval: `4`
- large leap ratio: `0.000`
- unresolved large leap ratio: `0.000`
- repeated pitch interval ratio: `0.000`

Variation candidates:

| candidate | notes | pitches | sync | bar-var | dur-var | ioi-var | ioi-rep | objective flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 59 | 25 | 0.683 | 0.750 | 0.095 | 0.065 | 0.484 | `{}` |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 59 | 24 | 0.762 | 0.500 | 0.079 | 0.081 | 0.371 | `{}` |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 63 | 25 | 0.641 | 0.500 | 0.078 | 0.063 | 0.381 | `{}` |

But it did not produce a `keep` candidate:

- rank 1 still has high IOI repetition: `0.484`
- rank 2 is the best sign, but low bar-position variation remains
- rank 3 has better note count and IOI repetition, but conservative diversity keeps phrase shape mechanical

## 해석

Issue #126 should not be reverted outright:

- it removes objective leap risk from variation review MIDI
- it keeps duplicate count at `0`
- it lowers dominant IOI repetition for two of three variation candidates

But it is not enough:

- all reviewed candidates remain `too_stiff`
- most candidates are still `fragment`
- aggregate still recommends phrase vocabulary first
- lower IOI repetition came with lower IOI/bar-position/duration diversity

## Decision

Issue #128 conclusion:

- timing-grid repair is useful as a guardrail, but not a musical solution.
- next generation work should target phrase-vocabulary diversity while preserving the objective-clean and duplicate-free properties.

Recommended next issue:

```text
Stage B rhythm variation phrase-vocabulary diversity repair
```

Target:

- restore bar-position/IOI/duration diversity without returning to high dominant IOI repetition
- add more phrase-level contour/call-response variation
- keep duplicate note sequence count at `0`
- keep objective flags `{}`, max interval bound, and final guide landing

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_timing_grid_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_timing_grid_repaired_codex_proxy/timing_grid_repaired_rhythm_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_timing_grid_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_timing_grid_repaired_codex_proxy/timing_grid_repaired_rhythm_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
