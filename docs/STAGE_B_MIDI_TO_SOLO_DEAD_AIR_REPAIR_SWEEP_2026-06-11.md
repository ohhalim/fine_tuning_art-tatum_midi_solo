# Stage B MIDI-to-Solo Dead-Air Repair Sweep

## Summary

- source case: `major_ii_v_turnaround`
- chords: `Dm7,G7,Cmaj7,A7`
- source strict: `2` / `6`
- best strict: `6` / `6`
- strict delta: `4`
- source dead-air fails: `4`
- best dead-air fails: `0`
- selected variant: `fill_n10`
- selected duration mode: `fill`
- selected note groups per bar: `10`
- rendered WAV files: `2`
- repair improved: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_repaired_progression_retry_sweep`

## Variants

| variant | duration | groups/bar | strict | grammar | dead-air fails | avg dead-air | avg removal | WAV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fill_n8` | `fill` | 8 | 4/6 | 6/6 | 2 | 0.7700 | 0.0938 | 2 |
| `fill_n9` | `fill` | 9 | 5/6 | 6/6 | 1 | 0.7616 | 0.1481 | 2 |
| `fill_n10` | `fill` | 10 | 6/6 | 6/6 | 0 | 0.6576 | 0.1750 | 2 |

## Best Variant WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_repair_sweep/issue_1222_dead_air_repair_sweep/packages/major_ii_v_turnaround_fill_n10/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_dead_air_repair_sweep/issue_1222_dead_air_repair_sweep/packages/major_ii_v_turnaround_fill_n10/audio/candidate_02_sample_02.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `full_progression_retry_sweep`
- `artist_level_long_solo_generation`
