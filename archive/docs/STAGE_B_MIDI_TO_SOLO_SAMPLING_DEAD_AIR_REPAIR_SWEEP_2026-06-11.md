# Stage B MIDI-to-Solo Dead-Air Repair Sweep

## Summary

- source case: `rhythm_turnaround`
- chords: `Bbmaj7,G7,Cm7,F7`
- source strict: `1` / `3`
- best strict: `3` / `3`
- strict delta: `2`
- source dead-air fails: `2`
- best dead-air fails: `0`
- selected variant: `fill_n9`
- selected duration mode: `fill`
- selected note groups per bar: `9`
- rendered WAV files: `2`
- repair improved: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_repaired_progression_retry_sweep`

## Variants

| variant | duration | groups/bar | strict | grammar | dead-air fails | avg dead-air | avg removal | WAV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fill_n8` | `fill` | 8 | 1/3 | 3/3 | 2 | 0.7854 | 0.1354 | 2 |
| `fill_n9` | `fill` | 9 | 3/3 | 3/3 | 0 | 0.6409 | 0.1481 | 2 |
| `fill_n10` | `fill` | 10 | 3/3 | 3/3 | 0 | 0.6701 | 0.1574 | 2 |

## Best Variant WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_repair_sweep/issue_1322_sampling_dead_air_repair/packages/rhythm_turnaround_fill_n9/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_repair_sweep/issue_1322_sampling_dead_air_repair/packages/rhythm_turnaround_fill_n9/audio/candidate_02_sample_01.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `full_progression_retry_sweep`
- `artist_level_long_solo_generation`
