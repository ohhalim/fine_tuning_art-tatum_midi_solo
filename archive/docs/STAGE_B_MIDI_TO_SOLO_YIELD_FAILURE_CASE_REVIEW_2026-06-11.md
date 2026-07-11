# Stage B MIDI-to-Solo Yield Failure Case Review

## Summary

- source strict yield: `18` / `24`
- min case strict yield rate: `0.3333`
- failing case count: `1`
- reviewed invalid samples: `4`
- dead-air failure count: `4`
- grammar primary cause excluded count: `1`
- collapse primary cause excluded count: `1`
- selected repair target: `duration_fill_or_overlap_aftercare`
- next boundary: `music_transformer_solo_yield_dead_air_repair_sweep`
- musical quality claimed: `false`

## Case Reviews

### major_ii_v_turnaround

- chords: `Dm7,G7,Cmaj7,A7`
- strict: `2` / `6`
- dominant failure: `dead_air_threshold_miss`
- repair target: `duration_fill_or_overlap_aftercare`
- invalid avg dead-air: `0.8568`
- strict avg dead-air: `0.7000`
- invalid avg postprocess removal: `0.1719`
- strict avg postprocess removal: `0.0938`

| sample | strict | failure | notes | dead air | removed | removal ratio | duration common | MIDI |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `true` | `none` | 16 | 0.7333 | 0 | 0.0000 | 1.0000 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_1.mid` |
| 2 | `false` | `dead-air ratio too high: 0.846 >= 0.800` | 14 | 0.8462 | 2 | 0.1250 | 0.8125 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_2.mid` |
| 3 | `true` | `none` | 13 | 0.6667 | 3 | 0.1875 | 0.8125 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_3.mid` |
| 4 | `false` | `dead-air ratio too high: 0.846 >= 0.800` | 14 | 0.8462 | 2 | 0.1250 | 0.9375 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_4.mid` |
| 5 | `false` | `dead-air ratio too high: 0.917 >= 0.800` | 13 | 0.9167 | 3 | 0.1875 | 1.0000 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_5.mid` |
| 6 | `false` | `dead-air ratio too high: 0.818 >= 0.800` | 12 | 0.8182 | 4 | 0.2500 | 0.8750 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/probes/02_major_ii_v_turnaround_seed_937/samples/stage_b_sample_6.mid` |

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `repair_effectiveness`
- `artist_level_long_solo_generation`
