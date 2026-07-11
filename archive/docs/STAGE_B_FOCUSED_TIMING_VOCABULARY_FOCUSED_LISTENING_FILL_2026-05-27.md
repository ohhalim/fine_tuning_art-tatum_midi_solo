# Stage B Focused Timing Vocabulary Focused Listening Fill

작성일: 2026-05-27

## 목적

Issue #194는 Issue #192 focused listening review notes template을 MIDI-note/context evidence 기준으로 채운 작업이다.

중요한 경계:

- 실제 오디오 청취 결과가 아니다.
- focused context MIDI-note 판단을 structured listening fields에 기록한 것이다.
- `needs_followup`은 objective-clean 후보가 쓸모없다는 뜻이 아니라, final keep으로 올리기 전 다음 repair 축이 남았다는 뜻이다.

## 입력

Focused listening notes template:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_focused_listening_notes/focused_listening_review_notes_template.json`

Filled notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_focused_listening_notes/focused_listening_review_notes_filled.json`

후보:

- `data_motif_rhythm_phrase_variation_rank_3_sample_3`

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
| landing | `strong` |
| jazz vocabulary | `thin` |
| decision | `needs_followup` |

Why not keep:

- The candidate survives focused context register and cadence checks.
- Solo range is usable at `G3-G5`.
- Final landing is a `D5` guide tone over `Ebmaj7`.
- There are no duplicated 4-note or 8-note pitch-class chunks.
- However timing is still grid-derived.
- Adjacent repeated pitches and duplicated 3-note cells remain.
- Source tension ratio is low at `0.297`, so jazz vocabulary still reads thin rather than conversational.

## Decision

Issue #194 conclusion:

- Do not promote the focused candidate to final keep.
- Keep it as a useful diagnostic seed.
- The next generation repair should target timing stiffness, short-cell vocabulary, and chord color while preserving:
  - objective-clean status
  - safe register range
  - final guide/chord landing
  - no overlap/polyphony
  - max interval guardrail
  - zero duplicated 4-note/8-note pitch-class chunks

Recommended next issue:

```text
Stage B focused timing vocabulary listening follow-up repair
```

Target:

- reduce grid-derived timing stiffness without creating off-grid artifacts
- reduce adjacent pitch repeats and duplicated 3-note pitch-class cells
- increase chord-color/tension usage without outside-note drift
- preserve the Issue #190 focused-context register/cadence guardrails
- re-run objective and proxy review before claiming quality

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_focused_listening_review_notes.py --run_id harness_stage_b_focused_timing_vocab_focused_listening_notes --focused_package outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.json --review_notes outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_focused_listening_notes/focused_listening_review_notes_filled.json
```
