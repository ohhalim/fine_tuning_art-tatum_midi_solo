# Stage B Register-Safe Timing Motif Repaired Proxy Review

작성일: 2026-05-27

## 목적

Issue #160은 Issue #158 register-safe timing motif follow-up repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, context MIDI path, objective MIDI metrics, first-note summary, Issue #156 focused listening fill 결과를 함께 본 proxy review다.
- objective-clean status는 최종 음악 품질이나 broad training 준비 완료를 의미하지 않는다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` baseline 후보 3개
- `data_motif_rhythm_phrase_variation` Issue #158 repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_timing_motif_repaired_codex_proxy/timing_motif_repaired_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_timing_motif_repaired_codex_proxy/listening_review_aggregate.md`

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
| phrase quality | `phrase=1`, `fragment=4`, `exercise=1` |
| timing | `too_stiff=6` |
| chord fit | `fits=6` |
| objective bucket | `clean=6` |
| objective flags | `{}` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | low-register contour artifact remains baseline failure evidence |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | usable final landing, but low-register cells and grid movement remain |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | context-compatible notes, but still mechanical contour fragment |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `too_stiff` | `fits` | `needs_followup` | best repaired variation, but not enough to override Issue #156 stiff/thin focused finding |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | objective-clean but still bounded scalar/chromatic cells |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `too_stiff` | `fits` | `needs_followup` | widest pitch count, but full-grid neighboring motion reads as exercise |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 16 | strongest remaining blocker |
| `fix_timing_grid` | 12 | timing surface still reads grid-derived |
| `increase_motif_variation` | 3 | repeated cells remain despite partial penalty repair |

## Best Candidate Judgment

Best repaired variation candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- decision: `needs_followup`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `63`
- unique pitch count: `18`
- pitch range: `G3-D#5`
- max active notes: `1`
- off-sixteenth-grid count: `0`
- stepwise interval ratio: `0.468`
- chromatic interval ratio: `0.274`
- chord-tone ratio: `0.508`
- tension ratio: `0.492`
- outside ratio: `0.000`

Why not keep:

- Issue #156 already exposed this candidate family as stiff/thin under focused review.
- Issue #158 mostly changed pitch-cell pressure, not the audible timing surface.
- the candidate is objective-clean but still relies on repeated bounded cells.
- unique pitch count remains thin for an 8-bar phrase.

## Decision

Issue #160 conclusion:

- Do not promote any repaired candidate to proxy `keep`.
- Keep the Issue #158 guard as a partial safety improvement, but do not keep pushing the same penalty-only approach.
- The next generation issue should target timing and phrase vocabulary from a data-derived phrase template, not another local cell-penalty tweak.
- Broad training, Brad style adaptation, backend/API, and audio pivot remain premature.

Recommended next issue:

```text
Stage B data-derived timing phrase vocabulary repair
```

Target:

- derive longer timing/phrase cells from real phrase templates or the existing motif catalog
- vary onset/duration cells at phrase level instead of only penalizing repeated pitch-class cells
- preserve register-safe bounds, max interval, final guide/chord landing, and objective-clean review output

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_timing_motif_repaired_codex_proxy --review_manifest outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json --objective_midi_review_report outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json --source_review_markdown outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_timing_motif_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_timing_motif_repaired_codex_proxy/timing_motif_repaired_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_timing_motif_repaired_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_timing_motif_repaired_codex_proxy/timing_motif_repaired_review_notes_codex_midi_proxy.json
RUN_ID=harness_stage_b_timing_motif_repaired_codex_proxy bash scripts/agent_harness.sh stage-b-listening-review-aggregate
bash scripts/agent_harness.sh quick
```
