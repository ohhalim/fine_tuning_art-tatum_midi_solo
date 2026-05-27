# Stage B Register-Safe Focused Listening Review Notes

작성일: 2026-05-27

## 목적

Issue #154는 Issue #152에서 `keep_for_focused_listening`으로 판단한 단일 후보를 실제 청취용 review notes template으로 만든 작업이다.

중요한 경계:

- 이 단계는 generation rule을 바꾸지 않는다.
- MIDI-note proxy decision과 실제 listening decision을 분리한다.
- 다음 generation 수정은 focused listening note가 채워진 뒤에만 판단한다.

## 구현

추가:

- `scripts/build_focused_listening_review_notes.py`
- `tests/test_focused_listening_review_notes.py`
- `bash scripts/agent_harness.sh stage-b-focused-listening-review-notes`

입력:

- `outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package/focused_review_package.json`

출력:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_listening_review_notes/focused_listening_review_notes_template.json`
- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_listening_review_notes/focused_listening_review_notes_summary.json`

## Template Result

Summary:

| field | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |

Candidate:

| candidate | proxy decision | proxy timing | notes | unique | tension | objective flags |
|---|---|---|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `keep` | `acceptable` | `63` | `18` | `0.349` | `[]` |

The template preserves:

- solo MIDI path
- context MIDI path
- source MIDI path
- source metrics
- proxy review decision and notes
- objective flags and objective bucket
- objective first 16 note summary
- pending real-listening fields

Real-listening fields:

- `timing`
- `chord_fit`
- `phrase_continuation`
- `landing`
- `jazz_vocabulary`
- `decision`
- free-form `notes`

## Decision

Issue #154 conclusion:

- The one-candidate focused listening review template is ready.
- The candidate remains pending until a real listening pass is filled.
- No generation repair should start from this artifact alone.

Recommended next issue:

```text
Stage B register-safe focused listening review fill
```

Target:

- listen to the solo/context MIDI pair
- fill the real-listening fields in the generated notes template
- decide whether the candidate remains `keep`, becomes `needs_followup`, or should be rejected
- only then choose the next generation or evaluation boundary

## 검증

실행한 검증:

```bash
.venv/bin/python -m unittest tests.test_focused_listening_review_notes
.venv/bin/python -m py_compile scripts/build_focused_listening_review_notes.py tests/test_focused_listening_review_notes.py
bash scripts/agent_harness.sh stage-b-focused-listening-review-notes
bash scripts/agent_harness.sh quick
```
