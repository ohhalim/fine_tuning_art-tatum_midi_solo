# Stage B Register-Safe Timing Motif Follow-Up Repair

작성일: 2026-05-27

## Summary

Issue #158은 Issue #156 focused listening fill에서 남은 blocker를 generation rule 쪽에서 좁게 다시 본 작업이다.

Issue #156 판단:

- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `weak`
- landing: `acceptable`
- jazz vocabulary: `thin`
- decision: `needs_followup`

이번 목표는 register-safe final cadence guardrail을 유지하면서 repeated pitch-class cell과 boxed-in phrase vocabulary를 줄이는 것이다.

중요한 경계:

- 이 결과는 실제 오디오 청취 리뷰가 아니다.
- MIDI note/context metric과 generated note sequence 기준의 objective repair다.
- broad training, Brad style adaptation, product MVP 성공 근거로 해석하지 않는다.

## Implementation

변경 대상:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`

남긴 변경:

- `register_safe_phrase_pitch_classes(...)`
  - recent phrase memory를 최근 `6`음에서 `8`음으로 확장했다.
  - development role에서 fresh pitch-class를 더 자주 앞으로 보내 repeated pitch-class cell을 덜 고르게 했다.
- `register_safe_phrase_cell_penalty(...)`
  - repeated cell lookback을 `18`에서 `32`로 확장했다.
  - 최근 4음 pitch class 반복, repeated 3-note cell, repeated 4-note cell, exact 4-note cell penalty를 강화했다.

제외한 변경:

- asymmetric timing-position variation을 시도했지만 IOI diversity와 repetition metric이 악화되어 최종 변경에서 제외했다.
- 따라서 이번 repair는 timing을 해결했다고 보지 않는다.
- 최종 변경은 motif/pitch-cell repetition을 줄이는 guard에 한정한다.

보존한 guardrail:

- focused-context register bounds
- max variation interval: `4`
- final cadence landing: guide/chord tone
- overlap-free solo-line review export
- objective MIDI flags: `{}`

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

`data_motif_rhythm_phrase_variation` summary:

| metric | value |
|---|---:|
| valid | `3/3` |
| strict | `3/3` |
| final landing resolved | `3/3` |
| max abs interval | `4` |
| avg syncopated onset ratio | `0.684` |
| avg unique bar-position pattern ratio | `0.958` |
| avg duration diversity ratio | `0.079` |
| avg most-common duration ratio | `0.384` |
| avg IOI diversity ratio | `0.091` |
| avg most-common IOI ratio | `0.385` |
| avg tension ratio | `0.358` |
| avg root-tone ratio | `0.021` |
| objective MIDI flag counts | `{}` |

Top variation candidates:

| candidate | notes | unique pitches | sync | bar-var | dur-var | ioi-var | ioi-rep | tension | max interval | landing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 63 | 18 | 0.667 | 1.000 | 0.079 | 0.097 | 0.371 | 0.381 | 4 | guide |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 63 | 17 | 0.683 | 1.000 | 0.079 | 0.097 | 0.419 | 0.349 | 4 | guide |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 64 | 19 | 0.703 | 0.875 | 0.078 | 0.079 | 0.365 | 0.344 | 4 | guide |

Objective MIDI note review for variation candidates:

| candidate | objective bucket | flags | note count | unique pitches | tension | outside |
|---|---|---|---:|---:|---:|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | `clean` | `[]` | 63 | 18 | 0.492 | 0.000 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | `clean` | `[]` | 63 | 17 | 0.476 | 0.000 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | `clean` | `[]` | 64 | 19 | 0.453 | 0.031 |

## Interpretation

Improved or preserved:

- strict/objective-clean status is preserved.
- final guide/chord landing stays resolved for all variation samples.
- max interval remains bounded at `4`.
- root-tone ratio remains low.
- the top candidate keeps the focused-context register-safe landing behavior.

Still blocking:

- timing stiffness is not solved.
- repeated pitch-class cells still remain at the phrase level.
- unique pitch count is still thin for an 8-bar line.
- this should not be promoted to a final `keep` without a fresh proxy/listening review.

Issue #158 conclusion:

- keep the register-safe phrase-cell penalty repair as a partial motif guard.
- do not claim the timing blocker is fixed.
- run a fresh proxy review of the repaired candidates before deciding whether this repair is musically better.

## Next Recommended Issue

```text
Stage B register-safe timing motif repaired proxy review
```

The next issue should fill MIDI-note/context proxy review notes for the repaired candidates and decide whether the partial motif guard is worth keeping, or whether a data-derived timing phrase vocabulary approach is needed.

## Validation

Executed:

```bash
.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
bash scripts/agent_harness.sh quick
```
