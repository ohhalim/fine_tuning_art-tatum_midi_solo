# Stage B Focused Timing Vocabulary Follow-up Focused Listening Fill

작성일: 2026-05-27

## Summary

Issue #208은 Issue #206 focused listening review notes template의 pending fields를 MIDI note/context evidence 기준으로 채운 작업이다.

Decision:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- filled decision: `keep`
- keep scope: current focused review candidate only
- next boundary: focused keep candidate consolidation

## Context

Issue #204는 이 후보를 `keep_for_focused_listening`으로 유지했다.

Issue #206은 후보의 solo/context MIDI path, proxy review, source metrics, objective first-note summary를 focused listening review notes template으로 분리했다.

이번 작업은 pending fields를 채워 다음 판단을 내리는 단계다.

중요한 경계:

- 이 판단은 MIDI note/context evidence 기준의 focused review fill이다.
- `keep`은 현재 후보가 다음 consolidation boundary로 넘어갈 수 있다는 뜻이다.
- `keep`은 final musical quality, broad training readiness, pianist style adaptation success를 의미하지 않는다.

## Inputs

Template:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_template.json`

Filled notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_filled.json`

Summary:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_summary.json`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

## Result

Validation summary:

| field | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `1` |
| pending count | `0` |
| keep | `1` |
| needs followup | `0` |
| reject | `0` |

Focused listening fields:

| field | value |
|---|---|
| timing | `acceptable` |
| chord fit | `strong` |
| phrase continuation | `acceptable` |
| landing | `strong` |
| jazz vocabulary | `acceptable` |
| decision | `keep` |

Supporting evidence:

- note count: `64`
- unique pitch count: `19`
- range: `G3-G5`
- phrase span: `32.0` beats
- final landing: `D5` over `Ebmaj7`
- final role: `guide/chord tone`
- max interval: `4`
- objective flags: `[]`
- adjacent pitch repeats: `0`
- adjacent pitch-class repeats: `0`
- duplicated 3-note pitch-class chunks: `0`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`
- source syncopated onset ratio: `0.719`
- source most-common IOI ratio: `0.397`
- objective chord-tone ratio: `0.516`
- objective tension ratio: `0.469`
- objective outside ratio: `0.016`

## Rationale

Why keep:

- Focused context checks still pass: register, context tracks, final landing, max interval, and objective flags have no blocker.
- The previous repeated-cell blocker is removed: adjacent pitch/pitch-class repeats are `0`, and duplicated 3/4/8-note pitch-class chunks are all `0`.
- The line covers the full 8-bar phrase span with `64` solo notes.
- Chord fit is strong at the objective level because chord-tone plus tension coverage is high and outside ratio is low.
- Timing is acceptable rather than strong: the line remains quantized, but syncopated onset ratio and duration variation avoid an immediate stiffness blocker.
- Jazz vocabulary is acceptable rather than strong: it has usable tension/color evidence, while `too_mechanical` remains a consolidation risk.

## Risk

Remaining risks:

- The proxy issue `too_mechanical` is not fully gone.
- Source IOI diversity remains low at `0.095`, with most-common IOI ratio `0.397`.
- The candidate is a single focused keep, not a multi-seed quality proof.
- This does not justify broad training, realtime integration, or style adaptation claims.

## Decision

Issue #208 conclusion:

- Keep the candidate as the current best focused review candidate.
- Stop doing local repair loops on this candidate until the keep evidence is consolidated.
- Move to a consolidation boundary that records what is now proven, what remains proxy-only, and what must be validated across more candidates before model-quality claims.

Recommended next issue:

```text
Stage B focused timing vocabulary keep candidate consolidation
```

Target:

- summarize the keep candidate as a reviewable MIDI outcome
- define the model-core MVP evidence now available
- list non-proven claims explicitly
- decide whether the next step is broader repeatability, candidate generalization, or portfolio README polish

## Validation

Executed validation:

```bash
.venv/bin/python scripts/build_focused_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_followup_focused_listening_notes --focused_package outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.json --review_notes outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_filled.json
bash scripts/agent_harness.sh quick
```
