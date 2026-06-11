# Stage B MIDI-to-Solo Chord Progression Yield Sweep

## Summary

- checkpoint generation used: `true`
- constrained decoding used: `true`
- case count: `4`
- sample count: `12`
- duration mode: `fill`
- valid yield: `8` / `12`
- strict yield: `8` / `12`
- grammar yield: `12` / `12`
- strict yield rate: `0.6667`
- min case strict yield rate: `0.3333`
- selected MIDI candidates: `0`
- rendered WAV files: `0`
- total yield floor passed: `false`
- all case yield floor passed: `false`
- musical quality claimed: `false`
- next boundary: `music_transformer_solo_yield_failure_case_review`

## Cases

| case | chords | seed | strict | valid | grammar | strict rate | avg notes | avg dead air | package |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `minor_backdoor` | `Cm7,F7,Bbmaj7,Ebmaj7` | 1600 | 3/3 | 3/3 | 3/3 | 1.0000 | 26.00 | 0.6946 | `none` |
| `major_ii_v_turnaround` | `Dm7,G7,Cmaj7,A7` | 1617 | 2/3 | 2/3 | 3/3 | 0.6667 | 28.50 | 0.7639 | `none` |
| `dominant_cycle` | `Em7,A7,Dmaj7,G7` | 1634 | 2/3 | 2/3 | 3/3 | 0.6667 | 28.50 | 0.6373 | `none` |
| `rhythm_turnaround` | `Bbmaj7,G7,Cm7,F7` | 1651 | 1/3 | 1/3 | 3/3 | 0.3333 | 26.00 | 0.7200 | `none` |

## Failing Cases

- `rhythm_turnaround`: strict `1` / `3`, rate `0.3333`

## WAV Files

- `none`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
