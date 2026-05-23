# Stage B Clean MIDI-Note Proxy Review

작성일: 2026-05-24

## 목적

Issue #109 clean review package, Issue #111 clean context diagnostics, Issue #113 clean listening review notes template 이후,
objective-clean context MIDI 후보 3개를 MIDI note-level로 다시 보고 follow-up 방향을 정리한다.

중요한 경계:

- 이 문서는 실제 오디오 청취 리뷰가 아니다.
- Codex가 MIDI note timing, pitch contour, context chord guide track을 읽어 작성한 piano-roll proxy review다.
- 따라서 `keep` 판단을 내리지 않고, 다음 generation probe 방향을 정하기 위한 보조 근거로만 사용한다.

## 입력

Review notes template:

```text
outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes/clean_listening_review_notes_template.json
```

Proxy-filled notes:

```text
outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes_codex_proxy/clean_listening_review_notes_codex_midi_proxy.json
```

대상 후보:

- `data_motif_phrase_recovery_rank_1_sample_1`
- `data_motif_phrase_recovery_rank_2_sample_2`
- `data_motif_phrase_recovery_rank_3_sample_3`

## Objective Context

세 후보 모두 objective gate 기준으로는 들을 수 있는 상태다.

| candidate | notes | unique | bars | off-grid | max duration | max simultaneous |
|---|---:|---:|---:|---:|---:|---:|
| `data_motif_phrase_recovery_rank_1_sample_1` | 63 | 19 | 8/8 | 0.000 | 1.000 | 1 |
| `data_motif_phrase_recovery_rank_2_sample_2` | 63 | 23 | 8/8 | 0.000 | 1.000 | 1 |
| `data_motif_phrase_recovery_rank_3_sample_3` | 63 | 22 | 8/8 | 0.000 | 1.000 | 1 |

이것은 성공 판정이 아니다.

의미는 다음과 같다.

- one-note/two-note collapse가 아니다.
- chord-block/polyphonic artifact가 아니다.
- long sustain block이 아니다.
- 8-bar coverage가 있다.
- context MIDI에 chord guide, bass root guide, solo track이 있다.

## Proxy Review Result

| candidate | timing | chord_fit | phrase_continuation | landing | jazz_vocabulary | decision |
|---|---|---|---|---|---|---|
| `data_motif_phrase_recovery_rank_1_sample_1` | `stiff` | `acceptable` | `acceptable` | `acceptable` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_2_sample_2` | `stiff` | `acceptable` | `weak` | `unresolved` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_3_sample_3` | `stiff` | `acceptable` | `broken` | `unresolved` | `exercise_like` | `reject` |

Aggregate:

- candidate count: `3`
- reviewed count: `3`
- pending count: `0`
- decisions:
  - `needs_followup`: `2`
  - `reject`: `1`
  - `keep`: `0`

## Candidate Notes

### Candidate 1

`data_motif_phrase_recovery_rank_1_sample_1`

Best of the three by MIDI-note proxy review.

Strengths:

- 8-bar coverage is complete.
- no simultaneous solo-note overlap.
- chord-tone and tension balance is close to even.
- final note lands on a chord tone over the last context chord.

Weaknesses:

- rhythm is fully grid-stiff.
- duration/rest template is nearly the same as the other candidates.
- contour still uses many large octave-like jumps instead of connected phrase resolution.
- jazz vocabulary reads as thin, not strong.

Decision:

```text
needs_followup
```

This is the candidate to preserve as the best current comparison point.

### Candidate 2

`data_motif_phrase_recovery_rank_2_sample_2`

Technically clean, but weaker phrase continuity than candidate 1.

Strengths:

- 8-bar coverage is complete.
- no simultaneous solo-note overlap.
- chord-tone ratio is slightly higher than candidate 1.

Weaknesses:

- bar 4 jumps into a high D6/F6 register area and then drops back sharply.
- the final note is C against the final context chord pitch classes `D`, `D#`, `G`, `A#`.
- the landing is unresolved by this proxy.
- phrase continuation feels more like register displacement than shaped melodic development.

Decision:

```text
needs_followup
```

Useful mainly for testing landing repair and contour smoothing.

### Candidate 3

`data_motif_phrase_recovery_rank_3_sample_3`

Reject by MIDI-note proxy review.

Strengths:

- objective MIDI gate is clean.
- 8-bar coverage is complete.
- no simultaneous solo-note overlap.

Weaknesses:

- bar 3 jumps up to the G#6/F6/C6 area.
- bar 4 collapses back toward F3/A3/A4.
- the register reset makes the phrase feel broken.
- final C does not resolve against the final context chord.
- grid-stiff rhythm plus extreme register movement reads as exercise-like.

Decision:

```text
reject
```

## Interpretation

The current best candidates are structurally valid, but not yet convincing jazz phrase candidates.

The problem is no longer basic MIDI validity.

Current blockers:

- rhythm stiffness
- repeated duration/rest template
- large register jumps without enough phrase-level contour control
- weak or unresolved final landing
- thin jazz vocabulary despite acceptable chord fit

This supports the direction already suggested by earlier docs:

- do not pivot to audio diffusion
- do not expand backend/UI scope
- do not move to broad training from these candidates
- do not add another hand-written rhythm patch without reference-driven justification

## Next Probe Direction

Recommended next issue:

```text
Stage B data-derived contour/cadence landing repair probe
```

The next probe should test:

- final-note landing repair against current or next chord guide tones
- register-contour smoothing after large leaps
- data-derived contour templates or cadence cells instead of one fixed grid pattern
- candidate 1 as the follow-up baseline
- candidate 3 as a negative example to avoid

Minimum success criteria:

- still no overlap/polyphonic solo-line artifact
- 8-bar coverage remains complete
- max simultaneous solo notes remains `1`
- final landing is not unresolved for top candidates
- contour avoids isolated high-register spikes followed by abrupt low-register reset
- rhythm/duration diversity improves without reintroducing duration collapse

## Validation

Proxy notes validation:

```bash
.venv/bin/python scripts/build_clean_listening_review_notes.py \
  --run_id harness_stage_b_clean_listening_review_notes_codex_proxy \
  --clean_package outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.json \
  --clean_context_diagnostics outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.json \
  --review_notes outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes_codex_proxy/clean_listening_review_notes_codex_midi_proxy.json
```

Result:

```json
{
  "candidate_count": 3,
  "reviewed_count": 3,
  "pending_count": 0,
  "decision_counts": {
    "reject": 1,
    "pending": 0,
    "keep": 0,
    "needs_followup": 2
  }
}
```
