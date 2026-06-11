# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `24`
- duration mode: `fill`
- valid yield: `20` / `24`
- strict yield: `20` / `24`
- grammar yield: `24` / `24`
- strict yield rate: `0.8333`
- min case strict yield rate: `0.6667`
- selected MIDI candidates: `8`
- rendered WAV files: `8`
- total yield floor passed: `true`
- all case yield floor passed: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_candidate_listening_review`

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 1500 | 4/6 | 4/6 | 6/6 | 0.6667 | 27.75 | 0.7571 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/01_minor_backdoor_seed_1500/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 1543 | 6/6 | 6/6 | 6/6 | 1.0000 | 27.33 | 0.7526 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/02_major_ii_v_turnaround_seed_1543/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 1586 | 5/6 | 5/6 | 6/6 | 0.8333 | 29.20 | 0.7171 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/03_dominant_cycle_seed_1586/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1629 | 5/6 | 5/6 | 6/6 | 0.8333 | 28.00 | 0.7337 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/04_rhythm_turnaround_seed_1629/solo_yield_package.json` |

## Failing Cases

- `none`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/01_minor_backdoor_seed_1500/audio/candidate_01_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/01_minor_backdoor_seed_1500/audio/candidate_02_sample_04.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/02_major_ii_v_turnaround_seed_1543/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/02_major_ii_v_turnaround_seed_1543/audio/candidate_02_sample_04.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/03_dominant_cycle_seed_1586/audio/candidate_01_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/03_dominant_cycle_seed_1586/audio/candidate_02_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/04_rhythm_turnaround_seed_1629/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1240_4bar_phrase_expansion_probe/packages/04_rhythm_turnaround_seed_1629/audio/candidate_02_sample_05.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
