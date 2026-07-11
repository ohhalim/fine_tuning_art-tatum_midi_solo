# Stage B MIDI-to-Solo Yield Failure Case Review

## Summary

- source strict yield: `8` / `12`
- min case strict yield rate: `0.3333`
- failing case count: `1`
- reviewed invalid samples: `2`
- dead-air failure count: `2`
- grammar primary cause excluded count: `1`
- collapse primary cause excluded count: `1`
- selected repair target: `duration_fill_or_overlap_aftercare`
- next boundary: `music_transformer_solo_yield_dead_air_repair_sweep`
- musical quality claimed: `false`

## Case Reviews

### rhythm_turnaround

- chords: `Bbmaj7,G7,Cm7,F7`
- strict: `1` / `3`
- dominant failure: `dead_air_threshold_miss`
- repair target: `duration_fill_or_overlap_aftercare`
- invalid avg dead-air: `0.8181`
- strict avg dead-air: `0.7200`
- invalid avg postprocess removal: `0.1094`
- strict avg postprocess removal: `0.1875`

| sample | strict | failure | notes | dead air | removed | removal ratio | duration common | MIDI |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `false` | `dead-air ratio too high: 0.821 >= 0.800` | 29 | 0.8214 | 3 | 0.0938 | 0.7188 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/probes/04_rhythm_turnaround_seed_1651/samples/stage_b_sample_1.mid` |
| 2 | `false` | `dead-air ratio too high: 0.815 >= 0.800` | 28 | 0.8148 | 4 | 0.1250 | 0.8125 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/probes/04_rhythm_turnaround_seed_1651/samples/stage_b_sample_2.mid` |
| 3 | `true` | `none` | 26 | 0.7200 | 6 | 0.1875 | 0.6250 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1318_sampling_repeatability_audit/probes/04_rhythm_turnaround_seed_1651/samples/stage_b_sample_3.mid` |

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `repair_effectiveness`
- `artist_level_long_solo_generation`
