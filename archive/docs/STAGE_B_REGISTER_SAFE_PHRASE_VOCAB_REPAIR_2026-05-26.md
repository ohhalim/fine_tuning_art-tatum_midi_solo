# Stage B Register-Safe Phrase Vocabulary Repair

작성일: 2026-05-26

## Summary

Issue #146는 Issue #144 proxy review에서 남은 boxed-in/cell-like phrase blocker를 generation rule 쪽에서 좁게 고친 작업이다.

목표는 다음 두 조건을 동시에 지키는 것이다.

- Issue #142의 focused-context register bounds와 final cadence safety를 유지한다.
- `data_motif_rhythm_phrase_variation`의 반복 셀을 줄여 phrase vocabulary를 조금 더 넓힌다.

이 결과는 여전히 symbolic MIDI probe 결과이며, broad training이나 Brad style adaptation 성공으로 해석하지 않는다.

## Implementation

변경 대상:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`

추가한 generation guard:

- `register_safe_phrase_pitch_classes(...)`
  - 기존 chord/guide/tension 후보 안에서 최근 6음 안에 나온 pitch class를 우선순위에서 밀고 fresh color를 앞으로 둔다.
  - 새 음역을 강제로 열지 않고 기존 register window 안에서만 작동한다.
- `register_safe_phrase_target_pitch(...)`
  - phrase center shaping 이후 작은 call/response offset을 추가한다.
  - bar-local register bounds로 clamp한다.
- `register_safe_phrase_cell_penalty(...)`
  - 이미 나온 3-4음 pitch-class cell 또는 exact 4-note cell을 다시 만들 때 선택 key에 penalty를 준다.
  - `data_motif_rhythm_phrase_variation`의 non-final interior note selection에만 적용했다.

보존한 guardrail:

- focused-context bounds: early bars `55-79`, final approach area `58-77`, final 2 bars `60-76`
- max variation interval: `4`
- final cadence landing: guide/chord tone
- duplicate review candidate detection

## Harness Result

Command:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Summary:

| Mode | valid | strict | final landing | max interval | objective flags |
| --- | ---: | ---: | ---: | ---: | --- |
| `data_motif_contour_landing_repair` | 3/3 | 3/3 | 3/3 | 7 | `{}` |
| `data_motif_rhythm_phrase_variation` | 3/3 | 3/3 | 3/3 | 4 | `{}` |

`data_motif_rhythm_phrase_variation` aggregate:

- `avg_unique_bar_position_pattern_ratio`: `0.958`
- `avg_ioi_diversity_ratio`: `0.091`
- `avg_most_common_ioi_ratio`: `0.385`
- `avg_tension_ratio`: `0.352`
- `avg_root_tone_ratio`: `0.032`
- `total_abrupt_register_reset_count`: `0`

Objective MIDI note review:

- candidate count: `6`
- flag counts: `{}`

## Candidate Notes

Top variation candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- note count: `63`
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- exact repeated 4-note cells in solo review MIDI: `0`
- duplicate note sequence: `false`

Variation candidate set:

| Candidate | unique pitches | max interval | exact repeated 4-note cells | duplicate note sequence |
| --- | ---: | ---: | ---: | --- |
| `data_motif_rhythm_phrase_variation_rank_1_sample_3` | 18 | 4 | 0 | false |
| `data_motif_rhythm_phrase_variation_rank_2_sample_1` | 17 | 4 | 0 | false |
| `data_motif_rhythm_phrase_variation_rank_3_sample_2` | 17 | 4 | 1 | false |

## Interpretation

Issue #146 keeps the Issue #142 register/cadence repair and reduces exact phrase-cell repetition in the highest-ranked repaired candidates.

The remaining blocker is musical quality, not objective MIDI validity:

- some pitch-class cells still repeat at the contour level
- timing is still grid-derived
- this should be judged with a fresh focused proxy/listening review before promoting any candidate back to `keep`

## Next Recommended Issue

```text
Stage B register-safe phrase vocabulary repaired proxy review
```

The next issue should fill proxy review notes for the repaired candidates and decide whether the phrase vocabulary repair is enough to restore a focused `keep` candidate.
