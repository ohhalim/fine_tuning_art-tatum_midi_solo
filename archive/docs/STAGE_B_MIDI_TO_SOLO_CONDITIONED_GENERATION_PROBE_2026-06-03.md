# Stage B MIDI-to-Solo Conditioned Generation Probe

## Summary

- boundary: `stage_b_midi_to_solo_conditioned_generation_probe`
- next boundary: `stage_b_midi_to_solo_candidate_audio_render_package`
- generation source: `context_conditioned_fallback`
- ranked MIDI candidates exported: `true`
- MIDI-to-solo MVP claimed: `false`
- human/audio preference claimed: `false`

## Input Context

- chord progression: `Cmaj7, F7, G7, Cmaj7, Cmaj7, Cmaj7, Cmaj7, Cmaj7`
- bars: `8`
- bpm: `120`
- density: `medium`

## Objective Gate

- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`

## Candidate Summary

- candidate count: `8`
- qualified candidate count: `8`
- exported candidate count: `3`
- exported qualified candidate count: `3`
- best score: `1.890847`
- best note count: `60`
- best unique pitch count: `14`
- best max simultaneous notes: `1`
- best chord-tone ratio: `1.0`

## Exported MIDI

- rank `1` seed `489`: `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_01_seed_489.mid`, score `1.890847`, notes `60`, unique pitches `14`
- rank `2` seed `488`: `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_02_seed_488.mid`, score `1.944383`, notes `59`, unique pitches `15`
- rank `3` seed `487`: `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_03_seed_487.mid`, score `1.984127`, notes `64`, unique pitches `13`

## Claim Boundary

- model checkpoint generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`
