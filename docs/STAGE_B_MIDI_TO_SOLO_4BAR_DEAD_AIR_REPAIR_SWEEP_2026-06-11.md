# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `24`
- duration mode: `fill`
- valid yield: `22` / `24`
- strict yield: `22` / `24`
- grammar yield: `24` / `24`
- strict yield rate: `0.9167`
- min case strict yield rate: `0.8333`
- selected MIDI candidates: `8`
- rendered WAV files: `8`
- total yield floor passed: `true`
- all case yield floor passed: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_candidate_listening_review`

## Repair Context

- source 4bar strict yield: `20` / `24`
- source 4bar grammar yield: `24` / `24`
- source 4bar case avg dead-air range: `0.7171` - `0.7571`
- repair variant: `note_groups_per_bar=9`, `max_sequence=160`
- repair strict yield: `22` / `24`
- repair grammar yield: `24` / `24`
- repair case avg dead-air range: `0.6340` - `0.6545`
- rejected variant: `note_groups_per_bar=10`, `max_sequence=192`
- rejected reason: checkpoint `model_max_sequence=160` positional encoding limit

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 1700 | 5/6 | 5/6 | 6/6 | 0.8333 | 31.80 | 0.6340 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/01_minor_backdoor_seed_1700/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 1743 | 6/6 | 6/6 | 6/6 | 1.0000 | 31.83 | 0.6545 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/02_major_ii_v_turnaround_seed_1743/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 1786 | 6/6 | 6/6 | 6/6 | 1.0000 | 31.00 | 0.6347 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/03_dominant_cycle_seed_1786/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1829 | 5/6 | 5/6 | 6/6 | 0.8333 | 30.00 | 0.6353 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/04_rhythm_turnaround_seed_1829/solo_yield_package.json` |

## Failing Cases

- `none`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/01_minor_backdoor_seed_1700/audio/candidate_01_sample_04.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/01_minor_backdoor_seed_1700/audio/candidate_02_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/02_major_ii_v_turnaround_seed_1743/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/02_major_ii_v_turnaround_seed_1743/audio/candidate_02_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/03_dominant_cycle_seed_1786/audio/candidate_01_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/03_dominant_cycle_seed_1786/audio/candidate_02_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/04_rhythm_turnaround_seed_1829/audio/candidate_01_sample_05.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1248_4bar_dead_air_repair_n9_seq160/packages/04_rhythm_turnaround_seed_1829/audio/candidate_02_sample_04.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
