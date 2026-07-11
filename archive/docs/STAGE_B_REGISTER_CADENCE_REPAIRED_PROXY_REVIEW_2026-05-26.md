# Stage B Register-Cadence Repaired Focused Proxy Review

작성일: 2026-05-26

## 목적

Issue #144는 Issue #142 register-cadence repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 focused proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 proxy review다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.
- broad training이나 style adaptation claim으로 해석하지 않는다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` register-cadence repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_register_cadence_repaired_codex_proxy/register_cadence_repaired_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_register_cadence_repaired_codex_proxy/listening_review_aggregate.md`

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 0 |
| `needs_followup` | 5 |
| `reject` | 1 |

Quality counts:

| field | result |
|---|---|
| phrase quality | `phrase=2`, `fragment=3`, `exercise=1` |
| timing | `acceptable=2`, `too_stiff=4` |
| chord fit | `fits=6` |
| objective bucket | `clean=6` |
| objective flags | `{}` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | C1-G2/D#4 bass-register artifact, final G2 |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | final G4 okay, but bars 3-6 still drop through G1-C3 cells |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register contour movement remains |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | register blocker repaired, but boxed-in/cell-like phrase remains |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | safe G3-G5 range and final G4, still stiff and repetitive |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `acceptable` | `fits` | `needs_followup` | objective-clean, but unique pitch count 15 and neighboring-tone exercise feel |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 14 | top blocker after register repair |
| `fix_timing_grid` | 8 | baseline and rank 2 still read too stiff |
| `increase_motif_variation` | 3 | variation candidates still repeat bounded cells |

## Repaired Top Candidate

`data_motif_rhythm_phrase_variation_rank_1_sample_3` is improved but not a final keep.

Positive evidence:

- objective flags: `[]`
- objective bucket: `clean`
- max active notes: `1`
- off-sixteenth-grid count: `0`
- note count: `63`
- pitch range: `C#4-G5`
- final landing: `G4`
- final bar notes: `F4, G4, A#4, A4, F4, D4, F#4, G4`

Blocking evidence:

- unique pitch count dropped to `18`
- stepwise/chromatic-heavy motion remains
- the line is safer than Issue #140, but still sounds boxed-in and cell-like by MIDI-note proxy review

## Decision

Issue #144 conclusion:

- Issue #142 register-cadence repair should be kept.
- The prior `C6` to final `G3` context blocker is fixed.
- No candidate is promoted to `keep`.
- The next repair should re-expand phrase vocabulary and motif development while keeping the new register bounds.

Recommended next issue:

```text
Stage B register-safe phrase vocabulary repair
```

Target:

- preserve Issue #142 register bounds and final cadence safety
- increase phrase vocabulary without returning to C6/G3 register arc
- reduce boxed-in neighboring-tone/cell repetition
- keep objective MIDI flags `{}`, duplicate note sequences `0`, and max interval guardrails

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_register_cadence_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_register_cadence_repaired_codex_proxy/register_cadence_repaired_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_register_cadence_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_register_cadence_repaired_codex_proxy/register_cadence_repaired_review_notes_codex_midi_proxy.json
bash scripts/agent_harness.sh quick
```

Quick harness result:

- unit tests: `236` passed
- compile checks: passed
- diff whitespace check: passed
