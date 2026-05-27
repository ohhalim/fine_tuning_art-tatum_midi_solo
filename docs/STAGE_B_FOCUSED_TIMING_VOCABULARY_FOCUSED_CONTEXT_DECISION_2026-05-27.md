# Stage B Focused Timing Vocabulary Focused Context Decision

작성일: 2026-05-27

## 목적

Issue #190은 Issue #188 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 focused proxy decision이다.
- `keep_for_focused_listening`은 focused listening review notes로 넘길 수 있다는 뜻이지, 최종 음악 품질 승인이나 broad training 준비 완료를 의미하지 않는다.

## 입력

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_3_sample_3`

MIDI:

- solo:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_03_sample_03_overlap_free.mid`
- context:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_03_sample_03_overlap_free_with_context.mid`

Context chord cycle:

```text
Cm7 | Fm7 | Bb7 | Ebmaj7 | Cm7 | Fm7 | Bb7 | Ebmaj7
```

## Positive Evidence

The candidate survives the focused-context register and cadence checks:

- objective flags: `[]`
- objective bucket: `clean`
- max active notes: `1`
- off-sixteenth-grid count: `0`
- note count: `64`
- unique pitch count: `20`
- pitch range: `G3-G5`
- final landing: `D5`
- final chord: `Ebmaj7`
- final role: `guide`
- max interval: `4`
- unresolved large leap ratio: `0.000`
- outside ratio: `0.000` in objective MIDI review
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`

Context track check:

- chord guide exists: `32` notes, range `C3-G#4`
- bass root guide exists: `8` notes, range `C2-A#2`
- solo track exists: `64` notes, range `G3-G5`

Bar-level MIDI-note role summary:

| bar | chord | notes | chord tones | tensions | outside |
|---:|---|---:|---:|---:|---:|
| 1 | `Cm7` | 9 | 5 | 2 | 2 |
| 2 | `Fm7` | 8 | 5 | 2 | 1 |
| 3 | `Bb7` | 8 | 4 | 2 | 2 |
| 4 | `Ebmaj7` | 8 | 5 | 3 | 0 |
| 5 | `Cm7` | 8 | 5 | 2 | 1 |
| 6 | `Fm7` | 8 | 5 | 2 | 1 |
| 7 | `Bb7` | 8 | 3 | 3 | 2 |
| 8 | `Ebmaj7` | 7 | 5 | 2 | 0 |

## Remaining Risk

Focused context decision keeps the candidate, but only as a listening-review candidate.

Remaining blockers:

- adjacent repeated pitch count is `2`.
- duplicated 3-note pitch-class chunks remain at `2`.
- timing still comes from a quantized grid; `acceptable` means the grid is not an immediate blocker, not that timing is natural.
- source tension ratio is `0.297`, so chord color may still feel conservative in listening review.
- phrase vocabulary still has a mechanical risk from short cell reuse.

## Decision

Focused context decision:

| field | value |
|---|---|
| prior proxy decision | `keep` |
| focused context decision | `keep_for_focused_listening` |
| keep as diagnostic seed | `yes` |
| ready for broad training | `no` |
| ready for style adaptation claim | `no` |

Issue #190 conclusion:

- The focused package does not show a register, cadence, or context-track blocker.
- The candidate is good enough to move into a focused listening review artifact.
- The candidate is still not proof of jazz quality, because timing/vocabulary naturalness remains unresolved.

Recommended next issue:

```text
Stage B focused timing vocabulary focused listening review notes
```

Target:

- create a one-candidate focused listening review notes artifact
- include solo/context MIDI paths and focused context decision fields
- record real-listening fields separately from MIDI-note proxy fields
- avoid changing generation rules until the focused listening note is filled

## 검증

실행한 검증:

```bash
bash scripts/agent_harness.sh quick
```
