# Stage B Register-Safe Proxy-Keep Focused Context Package

작성일: 2026-05-27

## 목적

Issue #150은 Issue #148에서 복구된 proxy `keep` 후보만 분리해 focused context review용 package로 묶은 작업이다.

중요한 경계:

- `keep`은 MIDI-note proxy 기준이다.
- 실제 오디오 청취 승인이나 최종 musical-quality claim이 아니다.
- broad training이나 Brad style adaptation으로 바로 확장하지 않는다.
- `outputs/` 아래 생성 artifact는 커밋하지 않는다.

## 구현

기존 도구:

- `scripts/build_focused_review_package.py`

Harness 보강:

- `scripts/agent_harness.sh`
  - `stage-b-proxy-keep-focused-package`가 `REVIEW_NOTES_FILE` 또는 `REVIEW_NOTES_PATH` 환경 변수를 받을 수 있게 했다.
  - 기존 Issue #138 기본 경로는 그대로 유지했다.
  - 이번 Issue #150의 register-safe proxy notes도 같은 harness로 재현할 수 있다.

## 입력

Review notes:

- `outputs/stage_b_listening_review_notes/harness_stage_b_register_safe_phrase_vocab_codex_proxy/register_safe_phrase_vocab_repaired_review_notes_codex_midi_proxy.json`

Objective MIDI review:

- `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`

## Package Result

생성 package:

- JSON:
  - `outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package/focused_review_package.json`
- Markdown:
  - `outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package/focused_review_package.md`
- copied solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free.mid`
- copied context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_register_safe_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_01_sample_03_overlap_free_with_context.mid`

Result:

| field | value |
|---|---:|
| decision filter | `keep` |
| candidate count | `1` |
| copied MIDI files | `2` |

Selected candidate:

| candidate | phrase | timing | chord fit | notes | unique pitches | source tension | objective flags |
|---|---|---|---|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `phrase` | `acceptable` | `fits` | `63` | `18` | `0.349` | `[]` |

Objective first-phrase note summary starts:

| start beats | duration beats | pitch |
|---:|---:|---|
| `0.00` | `0.25` | `A#4` |
| `0.25` | `0.50` | `G4` |
| `0.75` | `1.25` | `F4` |
| `2.00` | `0.25` | `G#4` |
| `2.25` | `0.50` | `A#4` |
| `2.75` | `0.50` | `D5` |
| `3.25` | `0.50` | `D#5` |
| `3.75` | `0.25` | `G5` |

## Decision

Issue #150 conclusion:

- The focused review package is reproducible from the Issue #148 structured review notes.
- Only the register-safe proxy `keep` candidate is included.
- This is now the correct single-candidate artifact for focused context review.
- The next decision should come from reviewing this one solo/context MIDI pair, not from another broad generation repair.

Recommended next issue:

```text
Stage B register-safe proxy-keep focused context decision
```

Target:

- review the copied solo/context MIDI pair
- record whether the proxy keep survives focused context review
- if it fails, classify the failure as timing, phrase vocabulary, register/contour, chord fit, or motif repetition
- if it passes, define the next narrow generation/evaluation boundary without claiming production readiness

## 검증

실행한 검증:

```bash
SOURCE_RUN_ID=harness_stage_b_register_safe_phrase_vocab_codex_proxy REVIEW_NOTES_FILE=register_safe_phrase_vocab_repaired_review_notes_codex_midi_proxy.json RUN_ID=harness_stage_b_register_safe_proxy_keep_focused_package bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
bash scripts/agent_harness.sh quick
```
