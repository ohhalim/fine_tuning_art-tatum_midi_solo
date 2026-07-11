# Stage B Focused Timing Vocabulary Follow-up Focused Context Decision

작성일: 2026-05-27

## Summary

Issue #204는 Issue #200 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Decision:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- focused context decision: `keep_for_focused_listening`
- next boundary: focused listening review notes

## Context

Issue #198 proxy review에서 `data_motif_rhythm_phrase_variation_rank_2_sample_2`가 유일한 `keep` 후보로 남았다.

Issue #200은 해당 후보만 focused review package로 분리했다.

이번 작업의 목적은 package 안의 solo/context MIDI를 다시 읽고, 다음 단계인 focused listening review notes로 넘길 만큼 register, cadence, repetition, context guide 조건을 유지하는지 판단하는 것이다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- objective/context MIDI-note 근거로 focused listening review에 넘길지 판단하는 단계다.
- `keep_for_focused_listening`은 최종 음악 품질 승인이나 broad training 준비 완료를 의미하지 않는다.

## Inputs

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.json`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

MIDI:

- solo:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- context:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`

Context chord cycle:

```text
Cm7 | Fm7 | Bb7 | Ebmaj7 | Cm7 | Fm7 | Bb7 | Ebmaj7
```

## Evidence

The candidate survives the focused-context structural checks.

Objective and solo-line metrics:

| metric | value |
|---|---:|
| objective bucket | `clean` |
| objective flags | `[]` |
| max active notes | `1` |
| off-sixteenth-grid count | `0` |
| note count | `64` |
| unique pitch count | `19` |
| pitch range | `G3-G5` |
| phrase span | `32.0` beats |
| duration range | `0.25-1.50` beats |
| max interval | `4` |
| unresolved large leap ratio | `0.000` |
| adjacent pitch repeats | `0` |
| adjacent pitch-class repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| duplicated 4-note pitch-class chunks | `0` |
| duplicated 8-note pitch-class chunks | `0` |
| objective chord-tone ratio | `0.516` |
| objective tension ratio | `0.469` |
| objective outside ratio | `0.016` |
| source syncopated onset ratio | `0.719` |
| source most-common IOI ratio | `0.397` |

Cadence:

| field | value |
|---|---|
| final note | `D5` |
| final start | `31.75` beats |
| final duration | `0.25` beats |
| final bar | `8` |
| final chord | `Ebmaj7` |
| final role | `guide/chord tone` |

Context track check:

| track | note count | range |
|---|---:|---|
| chord guide | `32` | `C3-G#4` |
| bass root guide | `8` | `C2-A#2` |
| solo | `64` | `G3-G5` |

Bar coverage:

| bar | chord | solo notes | first note | last note |
|---:|---|---:|---|---|
| 1 | `Cm7` | `9` | `D4` | `C4` |
| 2 | `Fm7` | `8` | `G#3` | `F#4` |
| 3 | `Bb7` | `8` | `G#4` | `A#4` |
| 4 | `Ebmaj7` | `8` | `A4` | `F5` |
| 5 | `Cm7` | `8` | `G5` | `D5` |
| 6 | `Fm7` | `8` | `A#4` | `F4` |
| 7 | `Bb7` | `8` | `F#4` | `A4` |
| 8 | `Ebmaj7` | `7` | `A#4` | `D5` |

## Decision

Focused context decision:

| field | value |
|---|---|
| prior proxy decision | `keep` |
| focused context decision | `keep_for_focused_listening` |
| keep as diagnostic seed | `yes` |
| ready for broad training | `no` |
| ready for style adaptation claim | `no` |

Rationale:

- No register, polyphony, or context-track blocker appears in the focused package.
- The final `D5` lands on the last `Ebmaj7` as a guide/chord tone.
- The repaired candidate keeps adjacent pitch repeats at `0`.
- The repaired candidate also keeps duplicated 3/4/8-note pitch-class chunks at `0`, which was the main follow-up target after the previous listening fill.
- The candidate is clean enough to move into a focused listening review notes artifact.

## Risk

Remaining risks:

- The proxy review still marked `too_mechanical` as an issue.
- Timing is `acceptable` only as a proxy field; it may still feel grid-derived in listening review.
- Source tension ratio is moderate at `0.344`; chord color may still read conservative.
- Source IOI diversity is still low, with most-common IOI ratio `0.397`.
- Focused context keep does not prove model quality or pianist style adaptation.

## Follow-up

Recommended next issue:

```text
Stage B focused timing vocabulary follow-up focused listening review notes
```

Target:

- create a one-candidate focused listening review notes artifact
- preserve solo/context MIDI paths and focused context decision fields
- keep real-listening fields separate from objective/context fields
- avoid another generation repair until the focused listening note is filled

## Validation

Executed validation:

```bash
bash scripts/agent_harness.sh quick
```
