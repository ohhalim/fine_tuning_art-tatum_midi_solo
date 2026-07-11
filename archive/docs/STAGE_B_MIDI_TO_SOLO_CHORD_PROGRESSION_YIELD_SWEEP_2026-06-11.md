# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `24`
- valid yield: `18` / `24`
- strict yield: `18` / `24`
- grammar yield: `24` / `24`
- strict yield rate: `0.7500`
- min case strict yield rate: `0.3333`
- selected MIDI candidates: `8`
- rendered WAV files: `8`
- total yield floor passed: `true`
- all case yield floor passed: `false`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_failure_case_review`

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 900 | 5/6 | 5/6 | 6/6 | 0.8333 | 14.60 | 0.7075 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/01_minor_backdoor_seed_900/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 937 | 2/6 | 2/6 | 6/6 | 0.3333 | 14.50 | 0.7000 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/02_major_ii_v_turnaround_seed_937/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 974 | 5/6 | 5/6 | 6/6 | 0.8333 | 14.60 | 0.7505 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/03_dominant_cycle_seed_974/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1011 | 6/6 | 6/6 | 6/6 | 1.0000 | 14.50 | 0.7393 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/04_rhythm_turnaround_seed_1011/solo_yield_package.json` |

## Failing Cases

- `major_ii_v_turnaround`: strict `2` / `6`, rate `0.3333`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/01_minor_backdoor_seed_900/audio/candidate_01_sample_04.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/01_minor_backdoor_seed_900/audio/candidate_02_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/02_major_ii_v_turnaround_seed_937/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/02_major_ii_v_turnaround_seed_937/audio/candidate_02_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/03_dominant_cycle_seed_974/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/03_dominant_cycle_seed_974/audio/candidate_02_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/04_rhythm_turnaround_seed_1011/audio/candidate_01_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1218_chord_progression_yield_sweep/packages/04_rhythm_turnaround_seed_1011/audio/candidate_02_sample_03.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
