# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `40`
- duration mode: `fill`
- valid yield: `40` / `40`
- strict yield: `40` / `40`
- grammar yield: `40` / `40`
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
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 2300 | 10/10 | 10/10 | 10/10 | 1.0000 | 31.00 | 0.6342 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/01_minor_backdoor_seed_2300/solo_yield_package.json` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 2329 | 10/10 | 10/10 | 10/10 | 1.0000 | 30.80 | 0.6543 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/02_major_ii_v_turnaround_seed_2329/solo_yield_package.json` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 2358 | 10/10 | 10/10 | 10/10 | 1.0000 | 31.20 | 0.6428 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/03_dominant_cycle_seed_2358/solo_yield_package.json` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 2387 | 10/10 | 10/10 | 10/10 | 1.0000 | 30.80 | 0.6655 | `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/04_rhythm_turnaround_seed_2387/solo_yield_package.json` |

## Failing Cases

- `none`

## WAV Files

- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/01_minor_backdoor_seed_2300/audio/candidate_01_sample_08.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/01_minor_backdoor_seed_2300/audio/candidate_02_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/02_major_ii_v_turnaround_seed_2329/audio/candidate_01_sample_07.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/02_major_ii_v_turnaround_seed_2329/audio/candidate_02_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/03_dominant_cycle_seed_2358/audio/candidate_01_sample_02.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/03_dominant_cycle_seed_2358/audio/candidate_02_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/04_rhythm_turnaround_seed_2387/audio/candidate_01_sample_03.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_sweep/issue_1344_broader_repaired_sampling_repeatability/packages/04_rhythm_turnaround_seed_2387/audio/candidate_02_sample_09.wav`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
