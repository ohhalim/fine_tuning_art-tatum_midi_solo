# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `12`
- duration mode: `fill`
- valid yield: `12` / `12`
- strict yield: `12` / `12`
- grammar yield: `12` / `12`
- strict yield rate: `1.0000`
- min case strict yield rate: `1.0000`
- selected MIDI candidates: `8`
- rendered WAV files: `8`
- total yield floor passed: `true`
- all case yield floor passed: `true`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_candidate_listening_review`

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 1600 | 3/3 | 3/3 | 3/3 | 1.0000 | 30.67 | 0.6857 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/01_minor_backdoor_seed_1600/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 1617 | 3/3 | 3/3 | 3/3 | 1.0000 | 31.00 | 0.6234 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/02_major_ii_v_turnaround_seed_1617/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 1634 | 3/3 | 3/3 | 3/3 | 1.0000 | 30.00 | 0.6906 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/03_dominant_cycle_seed_1634/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1651 | 3/3 | 3/3 | 3/3 | 1.0000 | 30.67 | 0.6409 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/04_rhythm_turnaround_seed_1651/solo_yield_package.json` |

## Failing Cases

- `none`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/01_minor_backdoor_seed_1600/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/01_minor_backdoor_seed_1600/audio/candidate_02_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/02_major_ii_v_turnaround_seed_1617/audio/candidate_01_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/02_major_ii_v_turnaround_seed_1617/audio/candidate_02_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/03_dominant_cycle_seed_1634/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/03_dominant_cycle_seed_1634/audio/candidate_02_sample_01.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/04_rhythm_turnaround_seed_1651/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1324_sampling_repaired_retry/packages/04_rhythm_turnaround_seed_1651/audio/candidate_02_sample_01.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
