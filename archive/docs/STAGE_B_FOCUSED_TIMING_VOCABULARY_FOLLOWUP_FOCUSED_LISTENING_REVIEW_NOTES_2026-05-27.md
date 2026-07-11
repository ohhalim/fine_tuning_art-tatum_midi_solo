# Stage B Focused Timing Vocabulary Follow-up Focused Listening Review Notes

작성일: 2026-05-27

## Summary

Issue #206은 Issue #204에서 `keep_for_focused_listening`으로 남은 단일 후보를 focused listening review notes template으로 분리한 작업이다.

Result:

- candidate count: `1`
- reviewed count: `0`
- pending count: `1`
- decision pending: `1`

## Context

Issue #204는 `data_motif_rhythm_phrase_variation_rank_2_sample_2`가 focused context에서 유지된다고 판단했다.

이번 작업은 실제 review fields를 채우기 전에 solo/context MIDI path, proxy review, objective first-note summary를 하나의 notes artifact에 보존하는 단계다.

중요한 경계:

- 이 문서는 실제 청취 결과를 채운 것이 아니다.
- real-listening fields는 모두 `pending`이다.
- pending notes만으로 generation rule을 바꾸지 않는다.
- `outputs/` 아래 generated notes는 커밋하지 않는다.

## Scope

포함한 작업:

- Issue #200 focused package를 입력으로 focused listening review notes template 생성
- single candidate의 solo/context/source MIDI path 보존
- proxy review, source metrics, objective first 16 notes 보존
- pending review field와 다음 fill boundary 문서화

범위 밖:

- 실제 청취 판단 채우기
- generation rule 변경
- final musical quality approval
- generated artifact commit

## Inputs

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.json`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

Focused context decision:

- `keep_for_focused_listening`

## Generated Artifact

Focused listening review notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_template.json`

Summary:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_summary.json`

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
- proxy phrase quality: `phrase`
- proxy timing: `acceptable`
- proxy chord fit: `fits`
- proxy issue: `too_mechanical`
- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `19`
- source tension ratio: `0.344`
- source syncopated onset ratio: `0.719`
- source most-common IOI ratio: `0.397`

Review files preserved:

- solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`
- source MIDI:
  - `outputs/stage_b_data_motif_compare/harness_stage_b_focused_timing_vocab_listening_followup_repair/samples/data_motif_rhythm_phrase_variation/data_motif_rhythm_phrase_variation_sample_2.mid`

Pending real-listening fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- decision

## Decision

Issue #206 conclusion:

- Focused listening notes template exists for the single focused-context keep candidate.
- No generation rule should be changed from this pending template alone.
- The next issue should fill the focused listening review notes using solo/context MIDI evidence.

## Risk

Remaining risks:

- The proxy review still carries `too_mechanical`.
- Timing may still feel grid-derived after real listening.
- Jazz vocabulary may remain thin despite the repeated-cell repair.
- A filled review can still downgrade the candidate to `needs_followup`.

## Follow-up

Recommended next issue:

```text
Stage B focused timing vocabulary follow-up focused listening review fill
```

Target:

- fill the real-listening fields for the one candidate
- keep final quality claims conservative
- if the candidate stays `keep`, define the next model-core boundary
- if it becomes `needs_followup`, classify whether the next repair should target timing, phrase vocabulary, motif variation, or chord color handling

## Validation

Executed validation:

```bash
FOCUSED_RUN_ID=harness_stage_b_focused_timing_vocab_followup_proxy_keep_package RUN_ID=harness_stage_b_focused_timing_vocab_followup_focused_listening_notes bash scripts/agent_harness.sh stage-b-focused-listening-review-notes
bash scripts/agent_harness.sh quick
```
