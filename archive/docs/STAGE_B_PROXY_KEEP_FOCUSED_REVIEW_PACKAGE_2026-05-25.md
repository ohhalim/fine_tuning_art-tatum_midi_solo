# Stage B Proxy-Keep Focused Review Package

작성일: 2026-05-25

## 목적

Issue #138은 Issue #136에서 처음 나온 proxy `keep` 후보만 분리해 focused context listening용 package로 묶은 작업이다.

중요한 경계:

- `keep`은 MIDI-note proxy 기준이다.
- 실제 오디오 청취 승인이나 최종 musical-quality claim이 아니다.
- broad training이나 Brad style adaptation으로 바로 확장하지 않는다.
- `outputs/` 아래 생성 artifact는 커밋하지 않는다.

## 구현

새 도구:

- `scripts/build_focused_review_package.py`

이 도구는 filled listening review notes에서 `listening.decision == keep` 후보만 선택한다.

패키지에 보존하는 정보:

- candidate id와 review metadata
- solo MIDI, context MIDI, source MIDI path
- note count, unique pitch count, timing/rhythm source metrics
- listening review decision/notes
- objective MIDI review flags/metrics
- objective first 16 notes summary

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
```

기본 입력:

- review notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_phrase_shape_tension_codex_proxy/phrase_shape_tension_repaired_review_notes_codex_midi_proxy.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`

## Package Result

생성 package:

- JSON:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/focused_review_package.json`
- Markdown:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/focused_review_package.md`
- copied solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free.mid`
- copied context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free_with_context.mid`

Result:

| field | value |
|---|---:|
| decision filter | `keep` |
| candidate count | `1` |
| copied MIDI files | `2` |

Selected candidate:

| candidate | phrase | timing | chord fit | notes | unique pitches | source tension | objective flags |
|---|---|---|---|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `63` | `28` | `0.413` | `[]` |

Objective first-phrase note summary starts:

| start beats | duration beats | pitch |
|---:|---:|---|
| `0.00` | `0.25` | `A#4` |
| `0.25` | `0.50` | `G#4` |
| `0.75` | `1.25` | `F4` |
| `2.00` | `0.25` | `G4` |
| `2.25` | `0.50` | `D#4` |
| `2.75` | `0.50` | `D4` |

## Decision

Issue #138 conclusion:

- The focused review package is now reproducible from structured review notes.
- Only the proxy `keep` candidate is included.
- This is the correct next artifact for focused context listening.
- The next decision should come from listening to this one candidate with its context MIDI, not from another broad generation repair.

Recommended next issue:

```text
Stage B proxy-keep focused context listening decision
```

Target:

- listen to the copied solo/context MIDI pair
- record whether the proxy keep survives real context listening
- if it fails, classify the failure as timing, phrase vocabulary, register/contour, chord fit, or motif repetition
- if it passes, define the next narrow generation/evaluation boundary without claiming production readiness

## 검증

실행한 검증:

```bash
.venv/bin/python -m py_compile scripts/build_focused_review_package.py tests/test_focused_review_package.py
.venv/bin/python -m unittest tests.test_focused_review_package
bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
bash scripts/agent_harness.sh quick
```

Quick harness result:

- unit tests: `234` passed
- compile checks: passed
- diff whitespace check: passed
