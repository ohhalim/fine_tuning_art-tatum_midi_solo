# Stage B Focused Timing Vocabulary Keep Candidate Consolidation

작성일: 2026-05-27

## Summary

Issue #210은 Issue #208에서 `keep`으로 남은 focused review candidate의 의미를 model-core MVP 관점에서 정리한 consolidation이다.

Candidate:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

Consolidated decision:

- This is the current best focused review candidate.
- It is a reviewable symbolic MIDI solo-line outcome under the current constrained Stage B pipeline.
- It is not broad model-quality proof, human listening proof, or pianist style adaptation proof.

## Context

The recent Stage B loop moved through these boundaries:

1. Issue #196 repaired immediate pitch-class reuse and fallback behavior.
2. Issue #198 selected `data_motif_rhythm_phrase_variation_rank_2_sample_2` as the proxy keep.
3. Issue #200 isolated the candidate into a focused package with solo/context MIDI.
4. Issue #204 kept the candidate for focused listening after context/register/cadence review.
5. Issue #206 created the one-candidate focused listening review notes template.
6. Issue #208 filled the notes and kept the candidate.

This consolidation exists to prevent the keep decision from being over-read.

## Candidate Evidence

Candidate metadata:

| field | value |
|---|---|
| candidate | `data_motif_rhythm_phrase_variation_rank_2_sample_2` |
| mode | `data_motif_rhythm_phrase_variation` |
| sample seed | `18` |
| review variant | `overlap_free_solo_line` |
| valid | `true` |
| strict valid | `true` |

MIDI and review artifacts:

- solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`
- focused notes:
  - `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_filled.json`

Structural evidence:

| metric | value |
|---|---:|
| note count | `64` |
| unique pitch count | `19` |
| range | `G3-G5` |
| phrase span | `32.0` beats |
| max active notes | `1` |
| max interval | `4` |
| objective flags | `[]` |
| adjacent pitch repeats | `0` |
| adjacent pitch-class repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| duplicated 4-note pitch-class chunks | `0` |
| duplicated 8-note pitch-class chunks | `0` |

Harmonic and cadence evidence:

| metric | value |
|---|---:|
| objective chord-tone ratio | `0.516` |
| objective tension ratio | `0.469` |
| objective outside ratio | `0.016` |
| final note | `D5` |
| final chord | `Ebmaj7` |
| final role | `guide/chord tone` |

Focused review fields:

| field | value |
|---|---|
| timing | `acceptable` |
| chord fit | `strong` |
| phrase continuation | `acceptable` |
| landing | `strong` |
| jazz vocabulary | `acceptable` |
| decision | `keep` |

## Proven

The current pipeline has proven the following within a focused constrained Stage B setting:

- A generated MIDI candidate can survive objective note-level review instead of only existing as a `.mid` file.
- The pipeline can preserve an overlap-free solo line with `64` notes over an 8-bar span.
- The latest repair removed the main repeated-cell blocker for the selected candidate.
- The candidate keeps register, final landing, max interval, context-track, and objective-clean guardrails.
- Review artifacts now preserve the full path from proxy review to focused context decision to focused listening fill.

## Not Proven

The current result does not prove:

- unconstrained model quality
- broad multi-seed pass-rate
- human/audio listening preference
- generic jazz pianist base quality
- Brad Mehldau style adaptation
- realtime DAW/plugin readiness
- backend/API/product MVP readiness

This distinction matters because the strongest current result is still a single focused candidate, not a distribution-level claim.

## Remaining Risk

Known risks:

- `too_mechanical` remains in the proxy review.
- Source IOI diversity remains low at `0.095`.
- Most-common IOI ratio remains `0.397`.
- Timing and jazz vocabulary are `acceptable`, not `strong`.
- The candidate could fail if the same criteria are applied to broader seeds, progressions, or less constrained generation.

## Decision

Issue #210 conclusion:

- Treat the candidate as a current best reviewable MIDI outcome.
- Stop local repair loops on this exact candidate until a broader reason appears.
- Use this result as portfolio evidence for the validation loop, not as a claim that the model is musically complete.

Recommended next issue:

```text
포트폴리오용 README 최종 정리
```

Alternative research next issue:

```text
Stage B focused timing vocabulary keep repeatability sweep
```

## Validation

Executed validation:

```bash
bash scripts/agent_harness.sh quick
```
