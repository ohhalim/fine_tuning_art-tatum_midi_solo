# Stage B MIDI-to-Solo Model-Direct Phrase Quality Diagnostics

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics`
- next boundary: `stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair`
- candidate count: `3`
- flag counts: `{'dead_air_gap': 3, 'wide_interval_contour': 3, 'wide_register_span': 3}`
- max interval max: `82`
- adjacent pitch repeat total: `0`
- max duration most-common ratio: `0.4166666666666667`
- max dead-air ratio: `0.6521739130434783`
- model-direct generation quality claimed: `false`
- human/audio preference claimed: `false`

## Candidate Diagnostics

| rank | notes | unique pitch | range | max interval | adjacent repeats | duration ratio | dead-air | flags |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 24 | 20 | 26-108 | 82 | 0 | 0.417 | 0.652 | wide_interval_contour, dead_air_gap, wide_register_span |
| 2 | 24 | 20 | 21-103 | 69 | 0 | 0.417 | 0.652 | wide_interval_contour, dead_air_gap, wide_register_span |
| 3 | 24 | 19 | 28-102 | 53 | 0 | 0.417 | 0.652 | wide_interval_contour, dead_air_gap, wide_register_span |

## Not Proven

- `model_direct_generation_quality`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
