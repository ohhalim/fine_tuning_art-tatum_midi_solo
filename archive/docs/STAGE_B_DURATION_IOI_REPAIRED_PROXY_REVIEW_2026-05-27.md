# Stage B Duration/IOI Repaired Proxy Review

작성일: 2026-05-27

## 목적

Issue #170은 Issue #168 duration/IOI objective repair 이후의 후보를 MIDI-note/context 기준으로 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, context MIDI path, objective MIDI metrics, Issue #168 metric tradeoff를 기준으로 판단했다.
- objective-clean status는 최종 음악 품질이나 broad training 준비 완료를 의미하지 않는다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_duration_ioi_objective_repair/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_duration_ioi_objective_repair/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_duration_ioi_proxy_review/duration_ioi_proxy_review_notes.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_duration_ioi_proxy_review/listening_review_aggregate.md`

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 0 |
| `needs_followup` | 4 |
| `reject` | 2 |

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
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | contour baseline remains low-register/grid fragment |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | context-compatible baseline, but not duration/IOI repair evidence |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `reject` | low-register contour material and prior timing profile |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `acceptable` | `fits` | `needs_followup` | best repaired surface, but cell-driven and thin pitch vocabulary |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | safe register and visible IOI gap vocabulary, but repeated small-cell contour |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `too_stiff` | `fits` | `needs_followup` | high stepwise ratio and most-common IOI repetition |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 12 | strongest remaining blocker |
| `fix_timing_grid` | 8 | IOI diversity rose, but timing is still grid-heavy |
| `increase_motif_variation` | 4 | repaired candidates still lean on small repeated cells |

## Best Candidate Judgment

Best repaired candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_1`
- decision: `needs_followup`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `16`
- syncopated onset ratio: `0.672`
- duration diversity ratio: `0.094`
- IOI diversity ratio: `0.111`
- most-common IOI ratio: `0.460`
- source tension ratio: `0.375`
- objective tension ratio: `0.484`
- final landing: `guide`
- max interval: `4`

Why not keep:

- IOI diversity improved, but the line still has a high most-common IOI ratio.
- unique pitch count is thin for an 8-bar phrase.
- stepwise/chromatic movement remains high enough to read as controlled exercise material.
- phrase vocabulary and motif variation remain stronger blockers than raw objective cleanliness.

## Decision

Issue #170 conclusion:

- Do not promote any candidate to proxy `keep`.
- Keep Issue #168 as an objective duration/IOI diversity repair, not as a final musical improvement.
- The next generation issue should target phrase vocabulary and motif variation while preserving objective-clean guardrails.
- Broad training, Brad style adaptation, backend/API, and audio pivot remain premature.

Recommended next issue:

```text
Stage B phrase vocabulary motif variation repair
```

Target:

- reduce small-cell mechanical contour
- keep Issue #168 IOI diversity improvement where possible
- reduce most-common IOI repetition rather than only increasing unique IOI count
- preserve register-safe bounds, max interval, final guide/chord landing, and objective-clean MIDI output

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_duration_ioi_proxy_review --review_manifest outputs/stage_b_data_motif_review/harness_stage_b_duration_ioi_objective_repair/review_manifest.json --objective_midi_review_report outputs/stage_b_objective_midi_review/harness_stage_b_duration_ioi_objective_repair/objective_midi_note_review.json --source_review_markdown outputs/stage_b_data_motif_review/harness_stage_b_duration_ioi_objective_repair/review_candidates.md
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_duration_ioi_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_duration_ioi_proxy_review/duration_ioi_proxy_review_notes.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_duration_ioi_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_duration_ioi_proxy_review/duration_ioi_proxy_review_notes.json
```
