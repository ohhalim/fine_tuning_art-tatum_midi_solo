# Stage B Phrase-Vocabulary Repaired Rhythm MIDI Proxy Review

작성일: 2026-05-25

## 목적

Issue #132는 Issue #130 phrase-vocabulary repaired rhythm 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, duplicate note-sequence fields를 기준으로 한 proxy review다.
- `keep` 후보를 선언하기보다 다음 generation rule 병목을 정하는 단계다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` phrase-vocabulary repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_repaired_codex_proxy/phrase_vocab_repaired_rhythm_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_vocab_repaired_codex_proxy/listening_review_aggregate.md`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 0 |
| `needs_followup` | 6 |
| `reject` | 0 |

Quality counts:

| field | result |
|---|---|
| phrase quality | `phrase=3`, `fragment=2`, `exercise=1` |
| timing | `acceptable=2`, `too_stiff=4` |
| chord fit | `fits=5`, `too_safe=1` |
| objective bucket | `clean=6` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register artifact remains |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest baseline, still rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register mechanical descent |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | best repaired candidate, still a phrase sketch |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | guardrails hold, but grid arc remains |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `acceptable` | `too_safe` | `needs_followup` | timing improves, but safe scalar exercise remains |

Aggregate follow-up signals:

| code | count | previous Issue #128 count | interpretation |
|---|---:|---:|---|
| `improve_phrase_vocabulary` | 11 | 14 | improved but still top blocker |
| `fix_timing_grid` | 8 | 12 | timing stiffness decreased, not solved |
| `increase_motif_variation` | 5 | 6 | repetition decreased modestly |
| `increase_tension_approach_vocabulary` | 2 | 0 | lower-tension repaired candidates need pitch-color follow-up |

## Phrase Repair Findings

Issue #130 improved the review surface:

- `too_stiff` decreased from `6` to `4`
- timing `acceptable` increased from `0` to `2`
- phrase count increased from `2` to `3`
- no duplicate note sequences
- objective MIDI flags remain `{}`
- variation review MIDI has max simultaneous notes `1`

Best repaired candidate:

| candidate | notes | pitches | bar-var | dur-rep | ioi-var | ioi-rep | objective flags |
|---|---:|---:|---:|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 63 | 28 | 1.000 | 0.381 | 0.097 | 0.371 | `[]` |

But it did not produce a `keep` candidate:

- rank 1 is less stiff by proxy, but still reads as a generated high-register phrase sketch
- rank 2 keeps the repaired guardrails but retains a predictable grid-shaped arc
- rank 3 has acceptable timing but reads as a safe scalar exercise

## Decision

Issue #132 conclusion:

- phrase-vocabulary repair should be kept.
- it improves timing proxy signals without breaking objective-clean/duplicate-free guardrails.
- it is not enough for a keep candidate.
- next generation work should target phrase shape and tension/approach vocabulary rather than reverting timing repair.

Recommended next issue:

```text
Stage B rhythm variation phrase-shape tension repair
```

Target:

- keep Issue #130 position/IOI guardrails
- improve phrase shape beyond high-register sketch or scalar exercise
- increase tension/approach color without introducing outside-note or unresolved-leap flags
- keep duplicate note sequence count `0`
- keep objective MIDI flags `{}`

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_phrase_vocab_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_repaired_codex_proxy/phrase_vocab_repaired_rhythm_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_phrase_vocab_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_repaired_codex_proxy/phrase_vocab_repaired_rhythm_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
