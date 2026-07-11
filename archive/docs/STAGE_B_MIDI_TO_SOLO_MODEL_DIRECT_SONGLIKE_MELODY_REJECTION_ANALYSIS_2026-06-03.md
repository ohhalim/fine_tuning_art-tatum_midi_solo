# Stage B MIDI-to-Solo Model-Direct Songlike Melody Rejection Analysis

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis`
- source boundary: `stage_b_midi_to_solo_model_direct_user_listening_review_fill`
- next boundary: `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision`
- candidate count: `3`
- uniform bar density count: `3`
- four-notes-per-bar template count: `3`
- duration template monotony count: `3`
- IOI template monotony count: `3`
- safe interval cap compression count: `3`
- four-bar rhythm cycle repeated count: `3`
- shared rhythm signature count: `3`
- max abs interval max: `9`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Candidate Signals

| rank | notes | bars | notes/bar mode | max interval | duration ratio | IOI ratio | flags |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 32 | 8 | 4 | 9 | 0.4375 | 0.4516 | `uniform_bar_density`, `four_notes_per_bar_template`, `duration_template_monotony`, `ioi_template_monotony`, `safe_interval_cap_compression`, `four_bar_rhythm_cycle_repeated` |
| 2 | 32 | 8 | 4 | 9 | 0.4375 | 0.4516 | `uniform_bar_density`, `four_notes_per_bar_template`, `duration_template_monotony`, `ioi_template_monotony`, `safe_interval_cap_compression`, `four_bar_rhythm_cycle_repeated` |
| 3 | 32 | 8 | 4 | 8 | 0.4375 | 0.4516 | `uniform_bar_density`, `four_notes_per_bar_template`, `duration_template_monotony`, `ioi_template_monotony`, `safe_interval_cap_compression`, `stepwise_contour_bias`, `four_bar_rhythm_cycle_repeated` |

## Not Proven

- `jazz_solo_musical_quality`
- `human_audio_keep_preference`
- `model_direct_generation_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
