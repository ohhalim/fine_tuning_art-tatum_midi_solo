# Music Transformer Solo Yield Objective Quality Rubric Baseline

## Summary

- candidate count: `8`
- quality proxy pass/fail: `3` / `5`
- major label counts: `dead_air_high=2, low_tension_color=2, weak_direction_change=2`
- watch label counts: `dead_air_watch=1, tension_high_watch=1`
- selected repair target: `dead_air_density_balance_repair`
- next boundary: `music_transformer_solo_yield_dead_air_density_balance_repair`
- critical user input required: `false`
- musical quality claimed: `false`

## Candidate Labels

| idx | case | rank | notes | dead air | direction | tension | major labels | watch labels | MIDI |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `minor_backdoor` | 1 | 30 | 0.7241 | 0.7059 | 0.1944 | `dead_air_high,low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_01_minor_backdoor_rank_01.mid` |
| 2 | `minor_backdoor` | 2 | 33 | 0.6250 | 0.4412 | 0.3611 | `weak_direction_change` | `tension_high_watch` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_02_minor_backdoor_rank_02.mid` |
| 3 | `major_ii_v_turnaround` | 1 | 28 | 0.6296 | 0.4848 | 0.2778 | `weak_direction_change` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_03_major_ii_v_turnaround_rank_01.mid` |
| 4 | `major_ii_v_turnaround` | 2 | 28 | 0.6667 | 0.6364 | 0.2500 | `none` | `dead_air_watch` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_04_major_ii_v_turnaround_rank_02.mid` |
| 5 | `dominant_cycle` | 1 | 31 | 0.6333 | 0.5882 | 0.1667 | `low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_05_dominant_cycle_rank_01.mid` |
| 6 | `dominant_cycle` | 2 | 30 | 0.6897 | 0.5294 | 0.2778 | `dead_air_high` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_06_dominant_cycle_rank_02.mid` |
| 7 | `rhythm_turnaround` | 1 | 30 | 0.6552 | 0.6471 | 0.2222 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_07_rhythm_turnaround_rank_01.mid` |
| 8 | `rhythm_turnaround` | 2 | 29 | 0.6429 | 0.6765 | 0.2500 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_listening_review/issue_1346_broader_repaired_listening_package/midi/candidate_08_rhythm_turnaround_rank_02.mid` |

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `rubric_threshold_calibration`
- `repair_effectiveness`
