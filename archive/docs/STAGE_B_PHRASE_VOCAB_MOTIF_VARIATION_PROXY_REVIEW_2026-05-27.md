# Stage B Phrase Vocabulary/Motif Variation Proxy Review

작성일: 2026-05-27

## 목적

Issue #174는 Issue #172 phrase vocabulary/motif variation repair 후보를 MIDI-note/context 기준으로 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, context MIDI path, objective MIDI metrics, Issue #172 metric tradeoff를 기준으로 판단했다.
- proxy `keep`은 final musical keep이 아니라 focused context review package로 넘길 수 있다는 뜻이다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_phrase_vocab_motif_variation_repair/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_phrase_vocab_motif_variation_repair/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_motif_variation_proxy_review/phrase_vocab_motif_variation_repaired_review_notes.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_vocab_motif_variation_proxy_review/listening_review_aggregate.json`

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
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | low-register contour drift and repeated grid feel |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `exercise` | `too_stiff` | `fits` | `needs_followup` | clean landing, but still low-register cell exercise |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `reject` | prior contour baseline problems remain |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `acceptable` | `fits` | `needs_followup` | reviewable range, but repeated cells and stiff bar grid remain |
| `data_motif_rhythm_phrase_variation_rank_2_sample_2` | `phrase` | `acceptable` | `fits` | `keep` | best proxy keep candidate for focused context review |
| `data_motif_rhythm_phrase_variation_rank_3_sample_3` | `phrase` | `acceptable` | `fits` | `needs_followup` | wide pitch vocabulary, but scalar/mechanical phrase feel remains |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 13 | still the strongest remaining blocker |
| `fix_timing_grid` | 6 | timing improved for repaired variation, but contour baseline remains stiff |
| `increase_motif_variation` | 3 | repeated cell feel remains in non-keep candidates |

## Proxy Keep Candidate

Selected candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- decision: `keep`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `18`
- source syncopated onset ratio: `0.719`
- source duration diversity ratio: `0.094`
- source most-common IOI ratio: `0.397`
- objective stepwise interval ratio: `0.460`
- objective tension ratio: `0.469`
- root-tone ratio: `0.063`
- final landing: `guide`
- max interval: `4`

Why keep only as proxy:

- It has the clearest mid-to-high phrase arc in the repaired set.
- It stays in a usable solo register and keeps final guide landing.
- It has the lowest stepwise ratio among the repaired variation candidates.
- It still has a mechanical duration/grid trace, so it is not final musical evidence.

## Decision

Issue #174 conclusion:

- Promote exactly one candidate to proxy `keep`.
- Do not claim final musical quality.
- The next issue should isolate the proxy keep candidate into a focused context review package.
- Broad training, Brad style adaptation, backend/API, and audio pivot remain premature.

Recommended next issue:

```text
Stage B phrase vocabulary motif proxy keep focused package
```

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_phrase_vocab_motif_variation_proxy_review --review_manifest outputs/stage_b_data_motif_review/harness_stage_b_phrase_vocab_motif_variation_repair/review_manifest.json --objective_midi_review_report outputs/stage_b_objective_midi_review/harness_stage_b_phrase_vocab_motif_variation_repair/objective_midi_note_review.json --source_review_markdown outputs/stage_b_data_motif_review/harness_stage_b_phrase_vocab_motif_variation_repair/review_candidates.md
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_phrase_vocab_motif_variation_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_motif_variation_proxy_review/phrase_vocab_motif_variation_repaired_review_notes.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_phrase_vocab_motif_variation_proxy_review --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_motif_variation_proxy_review/phrase_vocab_motif_variation_repaired_review_notes.json
```
