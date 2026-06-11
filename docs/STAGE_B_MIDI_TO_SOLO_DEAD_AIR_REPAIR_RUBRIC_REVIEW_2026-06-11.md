# Music Transformer Solo Yield Objective Quality Rubric Baseline

## Summary

- candidate count: `8`
- quality proxy pass/fail: `1` / `7`
- major label counts: `low_syncopation=1, low_tension_color=3, weak_direction_change=4`
- watch label counts: `none`
- selected repair target: `phrase_direction_balance_repair`
- next boundary: `music_transformer_solo_yield_phrase_direction_balance_repair`
- critical user input required: `false`
- musical quality claimed: `false`

## Candidate Labels

| idx | case | rank | notes | dead air | direction | tension | major labels | watch labels | MIDI |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `minor_backdoor` | 1 | 31 | 0.5333 | 0.4706 | 0.2222 | `weak_direction_change` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_01_minor_backdoor_sample_02.mid` |
| 2 | `minor_backdoor` | 2 | 32 | 0.5806 | 0.5000 | 0.1389 | `low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_02_minor_backdoor_sample_07.mid` |
| 3 | `major_ii_v_turnaround` | 3 | 31 | 0.5667 | 0.4118 | 0.2222 | `weak_direction_change` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_03_major_ii_v_turnaround_sample_02.mid` |
| 4 | `major_ii_v_turnaround` | 4 | 32 | 0.5806 | 0.4118 | 0.1667 | `weak_direction_change,low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_04_major_ii_v_turnaround_sample_05.mid` |
| 5 | `dominant_cycle` | 5 | 33 | 0.5625 | 0.5294 | 0.3056 | `none` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_05_dominant_cycle_sample_08.mid` |
| 6 | `dominant_cycle` | 6 | 31 | 0.6000 | 0.4242 | 0.3333 | `weak_direction_change` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_06_dominant_cycle_sample_05.mid` |
| 7 | `rhythm_turnaround` | 7 | 31 | 0.6000 | 0.6061 | 0.2778 | `low_syncopation` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_07_rhythm_turnaround_sample_05.mid` |
| 8 | `rhythm_turnaround` | 8 | 30 | 0.6207 | 0.5758 | 0.1944 | `low_tension_color` | `none` | `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_density_repair/issue_1370_dead_air_density_repair_package/midi/candidate_08_rhythm_turnaround_sample_08.mid` |

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `rubric_threshold_calibration`
- `repair_effectiveness`
