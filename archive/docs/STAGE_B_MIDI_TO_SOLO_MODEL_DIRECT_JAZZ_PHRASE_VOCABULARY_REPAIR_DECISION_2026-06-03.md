# Stage B MIDI-to-Solo Model-Direct Jazz Phrase Vocabulary Repair Decision

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision`
- source boundary: `stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis`
- next boundary: `stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe`
- candidate count: `3`
- shared rhythm signature count: `3`
- uniform bar density count: `3`
- four-notes-per-bar template count: `3`
- max abs interval max: `9`
- repair target count: `6`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Repair Targets

| target | current | target | acceptance signal |
|---|---|---|---|
| `break_uniform_bar_density` | `3` | `count <= 1` | `uniform_bar_density_count <= 1` |
| `replace_shared_rhythm_template` | `3` | `count <= 1` | `shared_rhythm_signature_count <= 1` |
| `reduce_duration_ioi_monotony` | `3/3` | `each count < 3` | `duration_template_monotony_count < candidate_count and ioi_template_monotony_count < candidate_count` |
| `restore_phrase_vocabulary` | `songlike_melody_not_soloing` | `phrase vocabulary candidate set` | `phrase_vocabulary_source != fixed_compact_template_only` |
| `relax_interval_cap_tradeoff` | `9` | `max interval <= 12 with controlled leap ratio` | `max_abs_interval_max <= 12 and safe_interval_cap_compression_count < candidate_count` |
| `preserve_objective_guards` | `timing repair strict candidates available` | `no overlap, bounded dead-air, bounded interval` | `no quality claim without a later listening package` |

## Not Proven

- `repair_probe_improves_listening_quality`
- `jazz_solo_musical_quality`
- `human_audio_keep_preference`
- `model_direct_generation_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
