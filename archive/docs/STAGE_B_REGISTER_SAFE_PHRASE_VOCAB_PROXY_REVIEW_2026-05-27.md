# Stage B Register-Safe Phrase Vocabulary Repaired Proxy Review

작성일: 2026-05-27

## 목적

Issue #148은 Issue #146 register-safe phrase vocabulary repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 proxy review다.
- `keep`은 focused context review로 넘길 proxy 후보라는 뜻이며, 최종 음악 품질이나 broad training 준비 완료를 의미하지 않는다.
- `outputs/` 아래 filled review notes와 aggregate는 생성 artifact라 커밋하지 않는다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` register-safe phrase vocabulary repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_register_safe_phrase_vocab_codex_proxy/register_safe_phrase_vocab_repaired_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_register_safe_phrase_vocab_codex_proxy/listening_review_aggregate.md`

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 1 |
| `needs_followup` | 4 |
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
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `reject` | C1-G2 bass-register artifact remains and final landing is G2 |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `fragment` | `too_stiff` | `fits` | `needs_followup` | final G4 is usable, but bars 3-6 still drop through low-register cells |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register contour movement and one unresolved large leap remain |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `keep` | proxy keep for focused context review only |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | safe final G4, but high stepwise/chromatic ratios still read as bounded scalar cells |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `acceptable` | `fits` | `needs_followup` | objective-clean, but low direction change and repeated pitch-class cells read like an exercise |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 13 | still the strongest remaining blocker |
| `fix_timing_grid` | 8 | baseline and rank 2 still read too stiff |
| `increase_motif_variation` | 3 | repeated pitch-class cells remain in variation candidates |

## Proxy Keep Candidate

`data_motif_rhythm_phrase_variation_rank_1_sample_3` is promoted back to proxy `keep` for focused context review.

Positive evidence:

- objective flags: `[]`
- objective bucket: `clean`
- note count: `63`
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- max active notes: `1`
- off-sixteenth-grid count: `0`
- outside ratio: `0.000`
- duplicate note sequence: `false`

Important limitations:

- repeated pitch-class cells still exist.
- timing is still grid-derived.
- unique pitch count remains lower than the earlier Issue #136 proxy keep candidate.
- this is not a real audio listening keep and not a broad-training claim.

## Decision

Issue #148 conclusion:

- Issue #146 register-safe phrase vocabulary repair should be kept.
- The top repaired variation candidate is strong enough to isolate as a focused context review candidate.
- Broad training and Brad style adaptation are still premature.
- The next issue should package the register-safe proxy keep candidate with context MIDI and objective summaries for a focused context decision.

Recommended next issue:

```text
Stage B register-safe proxy-keep focused context package
```

Target:

- copy only the proxy keep candidate and context MIDI into a focused package
- preserve objective metrics and MIDI-note summary
- keep comparison links back to the full review manifest
- do not claim final musical quality before focused context review

## 검증

실행한 검증:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_register_safe_phrase_vocab_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_register_safe_phrase_vocab_codex_proxy/register_safe_phrase_vocab_repaired_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_register_safe_phrase_vocab_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_register_safe_phrase_vocab_codex_proxy/register_safe_phrase_vocab_repaired_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
