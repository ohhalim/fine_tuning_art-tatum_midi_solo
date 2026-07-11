# Stage B Phrase-Shape Tension Repaired MIDI Proxy Review

작성일: 2026-05-25

## 목적

Issue #136은 Issue #134 phrase-shape/tension repaired rhythm 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- `keep`은 proxy 기준 focused context listening 후보라는 뜻이다.
- broad training이나 musical success claim으로 해석하지 않는다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` phrase-shape/tension repaired 후보 3개

Generated artifacts:

- review manifest:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_manifest.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_phrase_shape_tension_codex_proxy/phrase_shape_tension_repaired_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_shape_tension_codex_proxy/listening_review_aggregate.md`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 1 |
| `needs_followup` | 5 |
| `reject` | 0 |

Quality counts:

| field | result |
|---|---|
| phrase quality | `phrase=3`, `fragment=2`, `exercise=1` |
| timing | `acceptable=2`, `too_stiff=4` |
| chord fit | `fits=6` |
| objective bucket | `clean=6` |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register artifact remains |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest baseline, still rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | low-register mechanical descent |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `keep` | proxy keep for focused context listening |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | high-register grid arc remains |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `exercise` | `acceptable` | `fits` | `needs_followup` | exercise-like neighboring-tone pattern remains |

Aggregate follow-up signals:

| code | count | previous Issue #132 count | interpretation |
|---|---:|---:|---|
| `improve_phrase_vocabulary` | 10 | 11 | still top blocker, slightly improved |
| `fix_timing_grid` | 8 | 8 | timing issue remains mainly in baselines and rank 2 |
| `increase_motif_variation` | 5 | 5 | repetition still present outside the proxy keep |
| `increase_tension_approach_vocabulary` | 0 | 2 | tension repair addressed the too-safe proxy signal |

## Proxy Keep Candidate

`data_motif_rhythm_phrase_variation_rank_1_sample_3` is the first proxy keep candidate in this Stage B review chain.

Why it is kept by proxy:

- objective flags: `[]`
- duplicate note sequence: `false`
- note count: `63`
- unique pitch count: `28`
- max interval: `4`
- IOI repetition: `0.371`
- IOI diversity: `0.097`
- source tension ratio: `0.413`
- objective MIDI tension ratio: `0.540`
- first phrase now sits around D4-A#4 instead of staying in the high sketch register

Important limitation:

- this is not a real audio listening keep.
- it should be packaged for focused context listening before any training-scope expansion.

## Decision

Issue #136 conclusion:

- Issue #134 phrase-shape/tension repair should be kept.
- `data_motif_rhythm_phrase_variation_rank_1_sample_3` becomes a proxy keep candidate.
- broad training is still premature.
- next work should isolate the proxy keep candidate into a focused context review package.

Recommended next issue:

```text
Stage B proxy-keep rhythm candidate focused review package
```

Target:

- copy only the proxy keep candidate and context MIDI into a focused review package
- preserve objective metrics and first-note summaries
- keep comparison links back to the full manifest
- do not claim final musical quality without real listening

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_phrase_shape_tension_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_shape_tension_codex_proxy/phrase_shape_tension_repaired_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_phrase_shape_tension_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_phrase_shape_tension_codex_proxy/phrase_shape_tension_repaired_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
