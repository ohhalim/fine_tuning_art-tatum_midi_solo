# Stage B Phrase Vocabulary/Motif Focused Context Decision

작성일: 2026-05-27

## 목적

Issue #178은 Issue #176 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 focused proxy decision이다.
- `keep_for_focused_listening`은 focused listening review notes로 넘길 수 있다는 뜻이지, 최종 음악 품질 승인이나 broad training 준비 완료를 의미하지 않는다.

## 입력

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

MIDI:

- solo:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- context:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`

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
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- unresolved large leap ratio: `0.000`
- outside ratio: `0.000` in objective MIDI review
- adjacent repeated pitch count: `0`
- duplicated 8-note pitch-class chunks: `0`

Context track check:

- chord guide exists: `32` notes, range `C3-G#4`
- bass root guide exists: `8` notes, range `C2-A#2`
- solo track exists: `64` notes, range `G3-G5`

Bar-level MIDI-note role summary:

| bar | chord | notes | chord tones | tensions | outside |
|---:|---|---:|---:|---:|---:|
| 1 | `Cm7` | 9 | 5 | 3 | 1 |
| 2 | `Fm7` | 8 | 3 | 3 | 2 |
| 3 | `Bb7` | 8 | 5 | 2 | 1 |
| 4 | `Ebmaj7` | 8 | 4 | 4 | 0 |
| 5 | `Cm7` | 8 | 5 | 2 | 1 |
| 6 | `Fm7` | 8 | 5 | 2 | 1 |
| 7 | `Bb7` | 8 | 3 | 3 | 2 |
| 8 | `Ebmaj7` | 7 | 4 | 3 | 0 |

## Remaining Risk

Focused context decision keeps the candidate, but only as a listening-review candidate.

Remaining blockers:

- phrase vocabulary: `5` duplicated 3-note pitch-class chunks and `2` duplicated 4-note pitch-class chunks remain.
- timing: the line is still grid-derived; `timing=acceptable` means the grid is not an immediate blocker, not that it has natural jazz timing.
- motif variation: no duplicated 8-note chunk exists, but the shorter cells can still read mechanical.
- source tension ratio is only `0.344`, even though objective tension ratio is `0.469`.

## Decision

Focused context decision:

| field | value |
|---|---|
| prior proxy decision | `keep` |
| focused context decision | `keep_for_focused_listening` |
| keep as diagnostic seed | `yes` |
| ready for broad training | `no` |
| ready for style adaptation claim | `no` |

Issue #178 conclusion:

- The focused package does not show the prior low-register/context blocker.
- The candidate is good enough to move into a focused listening review artifact.
- The candidate is still not proof of jazz quality, because motif/timing naturalness remains unresolved.

Recommended next issue:

```text
Stage B phrase vocabulary motif focused listening review notes
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
