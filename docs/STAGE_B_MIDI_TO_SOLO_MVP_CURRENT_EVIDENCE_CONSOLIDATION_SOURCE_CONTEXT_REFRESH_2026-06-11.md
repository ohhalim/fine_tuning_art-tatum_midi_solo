# Stage B MIDI-to-Solo MVP Current Evidence Consolidation

## Summary

- boundary: `stage_b_midi_to_solo_mvp_current_evidence_consolidation`
- next boundary: `stage_b_midi_to_solo_readme_evidence_refresh`
- current MVP evidence supported: `true`
- technical execution evidence supported: `true`
- selected-scale objective path complete: `true`
- phrase-bank CLI technical path ready: `true`
- model-conditioned pitch-contour objective path ready: `true`
- model-conditioned pitch-contour changed-ratio repair objective path ready: `true`
- outside-soloing repair objective path ready: `true`
- outside-soloing repair source context preserved: `true`
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

## Phrase-Bank CLI Technical Path

- technical MIDI-to-solo CLI path ready: `true`
- explicit input used: `true`
- candidate / objective supported: `3` / `3`
- repaired MIDI / rendered WAV: `3` / `3`
- input context bars: `228`
- dead-air range: `0.1895-0.2211`
- preference fill allowed: `false`

## Model-Conditioned Pitch-Contour Objective Path

- current evidence consolidation ready: `true`
- rendered WAV files: `3`
- technical WAV validation: `true`
- max interval / threshold: `11` / `12`
- pitch-contour target supported: `true`
- max pitch changed ratio: `0.7174`
- pitch changed ratio review required: `true`
- audio review required: `true`
- preference fill allowed: `false`

## Model-Conditioned Pitch-Contour Changed-Ratio Repair Objective Path

- current evidence consolidation ready: `true`
- objective path supported: `true`
- rendered WAV files: `3`
- technical WAV validation: `true`
- max interval / threshold: `12` / `12`
- max pitch changed ratio / target: `0.4348` / `0.5000`
- changed-ratio target supported: `true`
- audio review required: `true`
- preference fill allowed: `false`

## Outside-Soloing Repair Objective Path

- current evidence consolidation ready: `true`
- objective path supported: `true`
- rendered WAV files: `6`
- technical WAV validation: `true`
- changed note total: `2`
- source objective outside-soloing pitch-role risk: `5`
- source outside-soloing pitch-role risk before / after / delta: `5` / `2` / `3`
- source outside-soloing repair targeted: `false`
- source outside-soloing residual risk preserved: `true`
- outside-soloing pitch-role risk after / delta: `0` / `2`
- follow-up objective source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- follow-up objective source outside-soloing source context preserved: `true`
- follow-up repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- follow-up repair sweep source outside-soloing source context preserved: `true`
- bridge repair sweep source outside-soloing source pitch-role risk: `5 -> 2`
- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `0 / 2`
- bridge repair sweep source outside-soloing source context preserved: `true`
- outside-soloing target supported: `true`
- weak landing target supported: `true`
- final landing chord-tone count after: `6`
- max non-chord-tone run after: `3`
- non-chord run target supported: `true`
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

- `Stage B MIDI-to-solo README evidence refresh source-context refresh`
