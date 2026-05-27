# Stage B Data-Derived Timing Phrase Repaired Proxy Review

작성일: 2026-05-27

## 목적

Issue #164는 Issue #162 data-derived timing phrase vocabulary repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, context MIDI path, objective MIDI metrics, Issue #160 proxy review 결과, Issue #162 metric tradeoff를 기준으로 판단했다.
- objective-clean status는 최종 음악 품질이나 broad training 준비 완료를 의미하지 않는다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_data_derived_timing_phrase_codex_proxy/data_derived_timing_phrase_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_data_derived_timing_phrase_codex_proxy/listening_review_aggregate.md`

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
| timing | `acceptable=2`, `too_stiff=4` |
| chord fit | `fits=6` |
| objective bucket | `clean=6` |
| objective flags | `{}` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | baseline low-register contour artifact |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | baseline contour fragment and grid movement |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | baseline context-compatible but not timing repair evidence |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | best repaired candidate, but still quantized/cell-driven |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low unique pitch count and high IOI repetition |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `acceptable` | `fits` | `needs_followup` | wider pitch count, but scalar/exercise-like contour |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 16 | strongest remaining blocker |
| `fix_timing_grid` | 8 | improved from Issue #160, but not solved |
| `increase_motif_variation` | 3 | repeated cells remain |

## Best Candidate Judgment

Best repaired candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- decision: `needs_followup`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `18`
- syncopated onset ratio: `0.672`
- duration diversity ratio: `0.078`
- most-common IOI ratio: `0.365`
- source tension ratio: `0.391`
- objective tension ratio: `0.484`
- final landing: `guide`
- max interval: `4`

Why not keep:

- it is the best repaired candidate, but not clearly enough beyond the Issue #156 focused blocker.
- the line remains quantized and cell-driven.
- unique pitch count stays thin for an 8-bar phrase.
- phrase vocabulary is still the top aggregate blocker.

## Decision

Issue #164 conclusion:

- Do not promote any candidate to proxy `keep`.
- Keep Issue #162 as a reviewable timing/tension tradeoff, not as a final quality improvement.
- The next generation issue should target phrase vocabulary with an explicit duration/IOI objective instead of another row-selection tweak.
- Broad training, Brad style adaptation, backend/API, and audio pivot remain premature.

Recommended next issue:

```text
Stage B phrase-level duration IOI objective repair
```

Target:

- preserve Issue #162 syncopation/tension gains where possible
- improve duration diversity and IOI diversity directly
- keep register-safe bounds, max interval, final guide/chord landing, and objective-clean MIDI output

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_data_derived_timing_phrase_codex_proxy --review_manifest outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json --objective_midi_review_report outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json --source_review_markdown outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_data_derived_timing_phrase_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_data_derived_timing_phrase_codex_proxy/data_derived_timing_phrase_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_data_derived_timing_phrase_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_data_derived_timing_phrase_codex_proxy/data_derived_timing_phrase_review_notes_codex_midi_proxy.json
RUN_ID=harness_stage_b_data_derived_timing_phrase_codex_proxy bash scripts/agent_harness.sh stage-b-listening-review-aggregate
bash scripts/agent_harness.sh quick
```
