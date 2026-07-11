# Stage B MIDI-to-Solo Phrase-Bank Retrieval Baseline

## Summary

- boundary: `stage_b_midi_to_solo_phrase_bank_retrieval_baseline`
- next boundary: `stage_b_midi_to_solo_phrase_bank_audio_render_package`
- generation source: `phrase_bank_data_motif_retrieval`
- ranked MIDI candidates exported: `true`
- MIDI-to-solo MVP claimed: `false`
- human/audio preference claimed: `false`

## Input Context

- chord progression: `Cmaj7, F7, G7, Cmaj7, Cmaj7, Cmaj7, Cmaj7, Cmaj7`
- bars: `8`
- bpm: `120`

## Phrase Bank

- source records: `56`
- motif count: `803`
- unique rhythm templates: `520`
- unique contour templates: `328`

## Objective Gate

- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`
- min phrase coverage ratio: `0.75`
- max dead-air ratio: `0.65`

## Candidate Summary

- candidate count: `9`
- qualified candidate count: `3`
- exported candidate count: `3`
- exported qualified candidate count: `3`
- best note count: `64`
- best unique pitch count: `22`
- best max simultaneous notes: `1`
- best dead-air ratio: `0.5873015873015873`
- best phrase coverage ratio: `1.0`

## Exported MIDI

- rank `1` mode `data_motif_rhythm_phrase_variation` seed `635`: `outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline/harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline/midi/rank_01_data-motif-rhythm-phrase-variation_seed_635.mid`, notes `64`, unique pitches `22`, dead-air `0.5873015873015873`
- rank `2` mode `data_motif_rhythm_phrase_variation` seed `632`: `outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline/harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline/midi/rank_02_data-motif-rhythm-phrase-variation_seed_632.mid`, notes `64`, unique pitches `21`, dead-air `0.5873015873015873`
- rank `3` mode `data_motif_rhythm_phrase_variation` seed `638`: `outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline/harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline/midi/rank_03_data-motif-rhythm-phrase-variation_seed_638.mid`, notes `64`, unique pitches `22`, dead-air `0.6031746031746031`

## Claim Boundary

- phrase-bank musical quality claimed: `false`
- model checkpoint generation quality claimed: `false`
- broad trained-model quality claimed: `false`
- Brad style adaptation claimed: `false`
