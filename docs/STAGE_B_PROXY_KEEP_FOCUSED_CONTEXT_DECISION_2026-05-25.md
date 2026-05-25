# Stage B Proxy-Keep Focused Context Decision

작성일: 2026-05-25

## 목적

Issue #140은 Issue #138 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 focused proxy decision이다.
- 이 결과만으로 broad training이나 style adaptation을 시작하지 않는다.

## 입력

Focused package:

- `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/focused_review_package.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`

MIDI:

- solo:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free.mid`
- context:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free_with_context.mid`

Context chord cycle:

```text
Cm7 | Fm7 | Bb7 | Ebmaj7 | Cm7 | Fm7 | Bb7 | Ebmaj7
```

## Positive Evidence

The candidate remains the strongest current Stage B rhythm-variation seed:

- objective flags: `[]`
- objective bucket: `clean`
- max active notes: `1`
- off-sixteenth-grid count: `0`
- note count: `63`
- unique pitch count: `28`
- max interval: `4`
- chord-tone ratio: `0.444`
- tension ratio: `0.540`
- outside ratio: `0.016`
- repeated pitch interval ratio: `0.000`
- no duplicated 4-note or 8-note pitch-class chunks in the solo note stream

Bar-level role summary after sixteenth-grid quantization:

| bar | chord | notes | chord tones | tensions | outside |
|---:|---|---:|---:|---:|---:|
| 1 | `Cm7` | 8 | 4 | 2 | 2 |
| 2 | `Fm7` | 8 | 3 | 3 | 2 |
| 3 | `Bb7` | 8 | 3 | 4 | 1 |
| 4 | `Ebmaj7` | 8 | 3 | 5 | 0 |
| 5 | `Cm7` | 8 | 4 | 3 | 1 |
| 6 | `Fm7` | 7 | 4 | 2 | 1 |
| 7 | `Bb7` | 8 | 3 | 4 | 1 |
| 8 | `Ebmaj7` | 8 | 4 | 3 | 1 |

## Blocking Evidence

Focused context decision downgrades this from proxy `keep` to `needs_followup`.

Primary blockers:

- register/contour: the line reaches `C6` around bar 4, then drifts down to `G3` by the final bar. The final two bars sit close to the bass/root guide register and read more like a low-register exercise ending than a confident solo-line cadence.
- phrase punctuation: the candidate fills most bars with similar eighth/quarter-grid cells. It has valid note counts, but the 8-bar arc still lacks clear phrase-level breathing and cadence punctuation.
- context fit: chord fit is objectively clean, but some outside notes are side-slip artifacts rather than clearly prepared/released color tones, especially early `G#4/E4` over `Cm7` and `D4/C#4` over `Fm7`.

This is a useful diagnostic seed, but it should not be promoted to a real listening pass yet.

## Decision

Focused context decision:

| field | value |
|---|---|
| prior proxy decision | `keep` |
| focused context decision | `needs_followup` |
| keep as diagnostic seed | `yes` |
| ready for broad training | `no` |
| ready for style adaptation claim | `no` |

Issue #140 conclusion:

- The Issue #138 package is useful and should stay as the focused review artifact.
- The candidate does not survive focused context MIDI-note review as a final keep.
- The next repair should preserve the objective-clean rhythm guardrails while adding register-arc control and cadence/phrase punctuation.

Recommended next issue:

```text
Stage B focused context register-arc cadence repair
```

Target:

- avoid ending the solo line in the bass-guide register
- keep final cadence in a right-hand solo register unless explicitly requested otherwise
- add phrase punctuation without reintroducing dead-air or overlap/polyphony flags
- preserve objective-clean status, duplicate-free status, and max interval guardrails

## 검증

실행한 검증:

```bash
bash scripts/agent_harness.sh quick
```

Quick harness result:

- unit tests: `234` passed
- compile checks: passed
- diff whitespace check: passed
