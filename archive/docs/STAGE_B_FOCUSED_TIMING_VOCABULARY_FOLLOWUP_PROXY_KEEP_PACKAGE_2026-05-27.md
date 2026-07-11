# Stage B Focused Timing Vocabulary Follow-up Proxy Keep Package

작성일: 2026-05-27

## Summary

Issue #200은 Issue #198 proxy keep 후보를 focused context review package로 분리한 작업이다.

## Context

Issue #198에서 `data_motif_rhythm_phrase_variation_rank_2_sample_2`가 유일한 proxy keep 후보로 남았다.

선정 근거:

- adjacent repeated pitch count: `0`
- duplicated 3-note pitch-class chunks: `0`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`
- pitch range: `G3-G5`
- final landing: `D5`
- max interval: `4`
- objective flags: `[]`

## Scope

포함한 작업:

- proxy keep 후보 1개만 focused package로 추출
- solo MIDI와 context MIDI를 package output으로 복사
- objective first-note summary와 proxy review metadata 보존
- 다음 focused context decision을 위한 입력 정리

범위 밖:

- 실제 오디오 청취 리뷰
- final quality approval
- broad training
- generated MIDI artifact commit

## Outputs

Generated package:

- package JSON:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.json`
- package markdown:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.md`
- copied solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- copied context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`

## Result

Focused package summary:

| field | value |
|---|---|
| decision filter | `keep` |
| candidate count | `1` |
| copied MIDI files | `2` |
| candidate | `data_motif_rhythm_phrase_variation_rank_2_sample_2` |
| mode | `data_motif_rhythm_phrase_variation` |
| sample seed | `18` |
| valid | `true` |
| strict valid | `true` |
| review variant | `overlap_free_solo_line` |

Candidate metrics:

| metric | value |
|---|---:|
| note count | `64` |
| unique pitch count | `19` |
| source syncopated onset ratio | `0.719` |
| source duration diversity ratio | `0.094` |
| source most-common duration ratio | `0.406` |
| source IOI diversity ratio | `0.095` |
| source most-common IOI ratio | `0.397` |
| source tension ratio | `0.344` |
| objective bucket | `clean` |
| objective flags | `[]` |

## Decision

Issue #200 conclusion:

- The proxy keep candidate is isolated as a one-candidate focused package.
- The package is ready for focused context MIDI-note decision.
- This remains a review input only, not a final musical-quality claim.

## Risk

- Proxy keep still carries `too_mechanical` from the proxy review.
- Source tension ratio is moderate at `0.344`.
- Focused context must still check phrase continuation, landing, context chord fit, and register behavior.

## Follow-up

Next issue:

```text
Stage B focused timing vocabulary follow-up focused context decision
```

Acceptance for the next boundary:

- inspect solo/context MIDI tracks
- verify register, final landing, max interval, repeated cell status, and context guide coverage
- decide whether to keep for focused listening notes or send back to repair

## Validation

Executed validation:

```bash
SOURCE_RUN_ID=harness_stage_b_focused_timing_vocab_followup_proxy_review OBJECTIVE_RUN_ID=harness_stage_b_focused_timing_vocab_listening_followup_repair REVIEW_NOTES_FILE=focused_timing_vocab_followup_repaired_review_notes.json RUN_ID=harness_stage_b_focused_timing_vocab_followup_proxy_keep_package bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
```
