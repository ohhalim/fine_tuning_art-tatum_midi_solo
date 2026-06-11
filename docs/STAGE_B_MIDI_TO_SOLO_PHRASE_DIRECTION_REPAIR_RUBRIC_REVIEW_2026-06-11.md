# Music Transformer Solo Yield Objective Quality Rubric Baseline

## Summary

- candidate count: `8`
- quality proxy pass/fail: `4` / `4`
- major label counts: `low_tension_color=4`
- watch label counts: `dead_air_watch=3`
- selected repair target: `tension_color_balance_repair`
- next boundary: `music_transformer_solo_yield_tension_color_balance_repair`
- critical user input required: `false`
- musical quality claimed: `false`

## Candidate Labels

| idx | case | rank | notes | dead air | direction | tension | major labels | watch labels | MIDI |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `minor_backdoor` | 1 | 30 | 0.6207 | 0.5294 | 0.2222 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_01_minor_backdoor_sample_01.mid` |
| 2 | `minor_backdoor` | 2 | 32 | 0.5806 | 0.5000 | 0.1389 | `low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_02_minor_backdoor_sample_07.mid` |
| 3 | `major_ii_v_turnaround` | 3 | 28 | 0.6667 | 0.6364 | 0.2500 | `none` | `dead_air_watch` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_03_major_ii_v_turnaround_sample_06.mid` |
| 4 | `major_ii_v_turnaround` | 4 | 34 | 0.6667 | 0.6176 | 0.1667 | `low_tension_color` | `dead_air_watch` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_04_major_ii_v_turnaround_sample_09.mid` |
| 5 | `dominant_cycle` | 5 | 31 | 0.6667 | 0.6176 | 0.1667 | `low_tension_color` | `dead_air_watch` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_05_dominant_cycle_sample_06.mid` |
| 6 | `dominant_cycle` | 6 | 31 | 0.6333 | 0.5882 | 0.1667 | `low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_06_dominant_cycle_sample_02.mid` |
| 7 | `rhythm_turnaround` | 7 | 29 | 0.6429 | 0.6765 | 0.2500 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_07_rhythm_turnaround_sample_09.mid` |
| 8 | `rhythm_turnaround` | 8 | 30 | 0.6552 | 0.6471 | 0.2222 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_phrase_direction_balance_repair/issue_1374_phrase_direction_balance_repair_package/midi/candidate_08_rhythm_turnaround_sample_03.mid` |

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `rubric_threshold_calibration`
- `repair_effectiveness`
