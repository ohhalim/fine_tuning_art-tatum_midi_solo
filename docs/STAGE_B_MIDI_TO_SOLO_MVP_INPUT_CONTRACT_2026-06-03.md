# Stage B MIDI-to-Solo MVP Input Contract

## Summary

- boundary: `stage_b_midi_to_solo_mvp_input_contract`
- next boundary: `stage_b_midi_to_solo_context_extraction_mvp`
- target date: `2026-06-11`
- user goal: `input_midi_to_jazz_solo_midi`
- success mode: `hybrid_model_constrained_decoder_candidate_ranking`
- neural-only required: `false`
- MIDI-to-solo MVP claimed: `false`
- Brad style fine-tuning completed: `false`

## Input Contract

- required input: `midi_file_path`
- supported extensions: `['.mid', '.midi']`
- minimum / target context bars: `4` / `8`
- chord confidence fallback: `low_confidence_context_allowed_but_ranked_lower`

## Output Contract

- output root: `outputs/stage_b_midi_to_solo_mvp`
- candidate count: `32`
- exported MIDI candidates: `3`
- target solo bars: `8`

## Generation Stack

- primary path: `generic_base_checkpoint_conditioned_generation`
- fallback path: `phrase_retrieval_data_motif_hybrid`
- fallback trigger: `zero_ranked_candidates_after_two_conditioned_generation_attempts`

## Objective Gate

- min note count: `24`
- min unique pitch count: `8`
- max dead-air ratio: `0.5`
- max long-note ratio: `0.5`
- max simultaneous notes: `1`
- max interval semitones: `12`
- min phrase coverage ratio: `0.75`

## Run Plan

- `2026-06-03` `stage_b_midi_to_solo_mvp_input_contract`: input/output contract and validation harness
- `2026-06-04` `stage_b_midi_to_solo_context_extraction_mvp`: MIDI context extractor MVP
- `2026-06-05` `stage_b_midi_to_solo_training_resource_probe`: near-full generic training resource probe
- `2026-06-06` `stage_b_midi_to_solo_conditioned_generation_probe`: conditioned generation candidate set
- `2026-06-07` `stage_b_midi_to_solo_constrained_decoder_ranking`: ranked MIDI candidates
- `2026-06-08` `stage_b_midi_to_solo_retrieval_fallback`: phrase retrieval fallback if needed
- `2026-06-09` `stage_b_midi_to_solo_cli_mvp`: generate_solo_from_midi CLI
- `2026-06-10` `stage_b_midi_to_solo_audio_review_package`: MIDI/WAV review package
- `2026-06-11` `stage_b_midi_to_solo_final_package`: README, usage guide, result and limitation boundary

## References

- MINGUS: jazz melodic line generation with chord, bass, position conditioning
- REMI / Pop Music Transformer: bar, position, chord and tempo context for symbolic piano generation
- Music Transformer: self-attention for continuation and motif-level structure
- Chord-progression phrase retrieval: fallback path when neural generation produces no ranked candidate
