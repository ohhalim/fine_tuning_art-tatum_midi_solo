# Stage B Clean Context Diagnostics

작성일: 2026-05-23

## Purpose

Issue #111은 Issue #109에서 추출한 objective-clean 후보 3개를 다시 MIDI note-level로 읽어, context listening review 전에 확인해야 할 구조적 위험을 정리하는 단계다.

이 단계는 새 generation rule을 추가하지 않는다. 목적은 다음 질문에 답하는 것이다.

- 후보가 실제로 8-bar phrase coverage를 갖는가?
- 박자가 grid에서 벗어난 timing drift인가?
- 긴 sustain이나 dominant pitch reuse가 다시 나타나는가?
- chord/bass/solo context MIDI가 함께 들어 있는가?
- 객관적으로는 들을 수 있는 후보인지, 아니면 다시 generation rule을 고쳐야 하는 후보인지?

## Implementation

Added:

- `scripts/build_clean_context_diagnostics.py`
- `tests/test_clean_context_diagnostics.py`
- `bash scripts/agent_harness.sh stage-b-clean-context-diagnostics`

Input:

- `outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.json`

Output:

- `outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.json`
- `outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.md`

## Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-clean-context-diagnostics
```

Summary:

- candidate count: `3`
- diagnostic flags: `{}`
- decision hints:
  - `listen_with_context`: `3`

Candidate metrics:

| candidate | notes | unique | bars | off-grid ratio | max duration beats | most-common pitch ratio | flags | hint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `data_motif_phrase_recovery_rank_1_sample_1` | 63 | 19 | 8/8 | 0.000 | 1.000 | 0.159 | none | `listen_with_context` |
| `data_motif_phrase_recovery_rank_2_sample_2` | 63 | 23 | 8/8 | 0.000 | 1.000 | 0.111 | none | `listen_with_context` |
| `data_motif_phrase_recovery_rank_3_sample_3` | 63 | 22 | 8/8 | 0.000 | 1.000 | 0.095 | none | `listen_with_context` |

Context summary:

- all candidates have context MIDI
- context MIDI contains chord guide
- context MIDI contains bass root guide
- context MIDI contains solo track

## Interpretation

The three candidates are not blocked by the objective failures that previously made review misleading.

They are:

- not sparse one-note/two-note outputs
- not low-coverage fragments
- not off-grid timing drift
- not long-sustain blocks
- not dominant-pitch collapse by the current thresholds
- packaged with chord/bass context

This still does not prove jazz quality.

It means the next correct step is listening review, not another automatic generation tweak.

## Next Decision

Listen to the context MIDI files and classify each candidate:

- timing: `good`, `too_loose`, `too_straight`, `unclear`
- chord fit: `fits`, `too_safe`, `too_outside`, `unclear`
- phrase continuation: `phrase_like`, `fragmented`, `exercise_like`, `unclear`
- landing: `clear`, `weak`, `missing`, `unclear`
- jazz vocabulary: `present`, `weak`, `absent`, `unclear`

If the candidates still sound like beginner chord-tone/tension enumeration, the next issue should target data-derived phrase/cadence vocabulary, not audio diffusion or backend expansion.
