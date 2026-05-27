# Stage B Phrase Vocabulary/Motif Focused Listening Fill

작성일: 2026-05-27

## 목적

Issue #182는 Issue #180 focused listening review notes template을 MIDI-note/context evidence 기준으로 채운 작업이다.

중요한 경계:

- 실제 오디오 청취 결과가 아니다.
- focused context MIDI-note 판단을 structured listening fields에 기록한 것이다.
- `needs_followup`은 objective-clean 후보가 쓸모없다는 뜻이 아니라, final keep으로 올리기 전 다음 repair 축이 남았다는 뜻이다.

## 입력

Focused listening notes template:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_template.json`

Filled notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_filled.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

## Filled Result

Validation summary:

| field | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `1` |
| pending count | `0` |
| keep | `0` |
| needs followup | `1` |
| reject | `0` |

Focused listening fields:

| field | value |
|---|---|
| timing | `stiff` |
| chord fit | `acceptable` |
| phrase continuation | `acceptable` |
| landing | `acceptable` |
| jazz vocabulary | `thin` |
| decision | `needs_followup` |

Why not keep:

- The candidate survives focused context register and cadence checks.
- Solo range is usable at `G3-G5`.
- Final landing remains `G4`.
- However timing is still grid-derived.
- Short pitch-class cells repeat enough to read mechanical.
- Jazz vocabulary is thin rather than conversational.

## Decision

Issue #182 conclusion:

- Do not promote the focused candidate to final keep.
- Keep it as a useful diagnostic seed.
- The next generation repair should target timing-grid stiffness and short pitch-cell vocabulary while preserving:
  - objective-clean status
  - safe register range
  - final guide/chord landing
  - no overlap/polyphony
  - max interval guardrail

Recommended next issue:

```text
Stage B focused timing vocabulary follow-up repair
```

Target:

- reduce grid-derived timing stiffness without creating off-grid artifacts
- reduce repeated 3-note/4-note pitch-class cells
- keep the Issue #178 focused-context register/cadence guardrails
- re-run objective and proxy review before claiming quality

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_focused_listening_review_notes.py --focused_package outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.json --run_id harness_stage_b_phrase_vocab_motif_focused_listening_notes --review_notes outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_filled.json
```
