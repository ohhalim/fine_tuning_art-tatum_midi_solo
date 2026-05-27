# Stage B Phrase Vocabulary/Motif Focused Listening Review Notes

작성일: 2026-05-27

## 목적

Issue #180은 Issue #178에서 `keep_for_focused_listening`으로 남은 단일 후보를 focused listening review notes template으로 만든 작업이다.

중요한 경계:

- 이 문서는 실제 청취 결과를 채운 것이 아니다.
- solo/context MIDI path, proxy review, objective first-note summary를 한 후보 notes 안에 보존한다.
- real-listening fields는 아직 `pending`이다.
- `outputs/` 아래 generated notes는 커밋하지 않는다.

## 입력

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

## Generated Artifact

Focused listening review notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_template.json`

Summary:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_summary.json`

## Result

Template summary:

| field | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| decision pending | `1` |

Candidate context:

- mode: `data_motif_rhythm_phrase_variation`
- sample seed: `18`
- valid: `true`
- strict valid: `true`
- review variant: `overlap_free_solo_line`
- proxy decision: `keep`
- proxy issue: `too_mechanical`
- objective bucket: `clean`
- objective flags: `[]`

Review files preserved:

- solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`

Pending real-listening fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- decision

## Decision

Issue #180 conclusion:

- Focused listening notes template exists for the single focused-context keep candidate.
- No generation rule should be changed from this pending template alone.
- Next step is to fill the focused listening review notes using solo/context MIDI evidence.

Recommended next issue:

```text
Stage B phrase vocabulary motif focused listening review fill
```

Target:

- fill the real-listening fields for the one candidate
- keep final quality claims conservative
- if the candidate stays `keep`, define the next model-core boundary
- if it becomes `needs_followup`, classify whether the next repair should target timing, phrase vocabulary, motif variation, or chord color handling

## 검증

실행한 검증:

```bash
FOCUSED_RUN_ID=harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package RUN_ID=harness_stage_b_phrase_vocab_motif_focused_listening_notes bash scripts/agent_harness.sh stage-b-focused-listening-review-notes
```
