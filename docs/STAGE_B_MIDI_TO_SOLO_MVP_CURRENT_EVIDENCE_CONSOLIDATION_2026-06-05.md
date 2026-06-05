# Stage B MIDI-to-Solo MVP Current Evidence Consolidation

## Summary

- boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- next boundary: `stage_b_midi_to_solo_readme_evidence_refresh`
- current MVP evidence supported: `true`
- technical execution evidence supported: `true`
- selected-scale objective path complete: `true`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Input Contract

- candidate count: `32`
- exported MIDI candidates: `3`
- target solo bars: `8`
- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`
- fallback path: `phrase_retrieval_data_motif_hybrid`

## Context and Ranked MIDI

- context bars / events: `8` / `128`
- low-confidence chord bars: `4`
- generation source: `context_conditioned_fallback`
- exported / qualified candidates: `3` / `3`
- best note / unique pitch / max simultaneous notes: `60` / `14` / `1`

## Technical Audio Path

- rendered WAV files: `3`
- sample rate: `44100`
- duration range: `18.617s-18.991s`
- technical WAV validation: `true`

## Selected-Scale Objective Path

- final boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_path_complete`
- sample / seed count: `9` / `3`
- valid / strict / grammar: `9` / `9` / `9`
- dead-air / collapse failure count: `0` / `0`
- avg / max postprocess removal ratio: `0.21759259259259262` / `0.2916666666666667`
- target avg postprocess removal ratio: `0.3`
- validated review input present: `false`
- preference fill allowed: `false`

## MIDI Paths

- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_01_seed_489.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_02_seed_488.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_03_seed_487.mid`

## WAV Paths

- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_01_seed_489.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_02_seed_488.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_03_seed_487.wav`

## Not Proven

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
- `production_ready_improviser`

## Next

- `Stage B MIDI-to-solo README evidence refresh`
