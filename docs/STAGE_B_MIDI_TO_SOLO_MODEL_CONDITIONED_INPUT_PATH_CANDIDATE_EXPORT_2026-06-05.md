# Stage B MIDI-to-Solo Model-Conditioned Input Path Candidate Export

## Summary

- boundary: `stage_b_midi_to_solo_model_conditioned_input_path_candidate_export`
- next boundary: `stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package`
- generation source: `model_checkpoint_direct_constrained`
- ranked MIDI candidates exported: `true`
- ranked input-path export contract matched: `true`
- fallback replacement candidate export ready: `true`
- fallback replacement ready: `false`
- candidate audio render required: `true`

## Probe Source

- phrase-bank CLI technical path completed: `true`
- CLI candidate / rendered WAV: `3` / `3`
- CLI input context bars: `228`
- CLI preference fill allowed: `false`

## Input Context

- chord progression: `Cmaj7, F7, G7, Cmaj7, Cmaj7, Cmaj7, Cmaj7, Cmaj7`
- bars: `8`
- bpm: `120`

## Candidate Summary

- candidate count: `3`
- exported candidate count: `3`
- exported qualified candidate count: `3`
- best score: `30.528061`
- best note count: `24`
- best unique pitch count: `20`
- best max simultaneous notes: `1`
- best chord-tone ratio: `0.6666666666666666`
- best dead-air ratio: `0.6521739130434783`

## Exported MIDI

- rank `1` sample `1` seed `497`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/midi/rank_01_sample_01.mid`, score `30.528061`, notes `24`, unique pitches `20`
- rank `2` sample `2` seed `498`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/midi/rank_02_sample_02.mid`, score `31.779061`, notes `24`, unique pitches `20`
- rank `3` sample `3` seed `499`: `outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/midi/rank_03_sample_03.mid`, score `31.988394`, notes `24`, unique pitches `19`

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- model checkpoint generation quality claimed: `false`
- broad trained-model quality claimed: `false`

## Next

- `Stage B MIDI-to-solo model-conditioned input path audio render package`
