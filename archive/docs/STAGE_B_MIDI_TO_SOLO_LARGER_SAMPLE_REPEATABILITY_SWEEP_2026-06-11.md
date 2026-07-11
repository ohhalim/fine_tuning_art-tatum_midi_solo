# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `48`
- duration mode: `fill`
- valid yield: `47` / `48`
- strict yield: `47` / `48`
- grammar yield: `48` / `48`
- strict yield rate: `0.9792`
- min case strict yield rate: `0.9167`
- selected MIDI candidates: `12`
- rendered WAV files: `12`
- total yield floor passed: `true`
- all case yield floor passed: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_candidate_listening_review`

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 1200 | 11/12 | 11/12 | 12/12 | 0.9167 | 16.55 | 0.5885 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/01_minor_backdoor_seed_1200/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 1241 | 12/12 | 12/12 | 12/12 | 1.0000 | 17.42 | 0.6102 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/02_major_ii_v_turnaround_seed_1241/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 1282 | 12/12 | 12/12 | 12/12 | 1.0000 | 16.83 | 0.6089 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/03_dominant_cycle_seed_1282/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1323 | 12/12 | 12/12 | 12/12 | 1.0000 | 16.50 | 0.6349 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/04_rhythm_turnaround_seed_1323/solo_yield_package.json` |

## Failing Cases

- `none`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/01_minor_backdoor_seed_1200/audio/candidate_01_sample_11.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/01_minor_backdoor_seed_1200/audio/candidate_02_sample_07.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/01_minor_backdoor_seed_1200/audio/candidate_03_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/02_major_ii_v_turnaround_seed_1241/audio/candidate_01_sample_10.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/02_major_ii_v_turnaround_seed_1241/audio/candidate_02_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/02_major_ii_v_turnaround_seed_1241/audio/candidate_03_sample_07.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/03_dominant_cycle_seed_1282/audio/candidate_01_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/03_dominant_cycle_seed_1282/audio/candidate_02_sample_05.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/03_dominant_cycle_seed_1282/audio/candidate_03_sample_07.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/04_rhythm_turnaround_seed_1323/audio/candidate_01_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/04_rhythm_turnaround_seed_1323/audio/candidate_02_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1232_fill_n10_sample12_repeatability/packages/04_rhythm_turnaround_seed_1323/audio/candidate_03_sample_12.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
