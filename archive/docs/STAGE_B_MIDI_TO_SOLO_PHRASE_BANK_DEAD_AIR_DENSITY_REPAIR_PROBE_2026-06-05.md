# Stage B MIDI-to-Solo Phrase-Bank Dead-Air Density Repair Probe

## Summary

- boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe`
- source boundary: `stage_b_midi_to_solo_phrase_bank_objective_only_next_decision`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package`
- repaired candidate count: `3`
- qualified repaired candidate count: `3`
- repair probe target passed: `true`
- repaired dead-air range: `0.1895 - 0.2211`
- dead-air gain range: `0.3768 - 0.3978`
- additions per bar target: `[3, 5, 2, 6, 3, 5, 2, 6]`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Candidate Repair Review

### Rank 1

- seed: `635`
- source dead-air: `0.5873`
- repaired dead-air: `0.1895`
- dead-air gain: `0.3978`
- note count gain: `32`
- unique density values: `4`
- duration diversity / IOI diversity: `0.0833 / 0.0632`
- qualified: `true`
- flags: ``
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/midi/rank_01_seed_635_dead_air_density_repair.mid`

### Rank 2

- seed: `632`
- source dead-air: `0.5873`
- repaired dead-air: `0.2105`
- dead-air gain: `0.3768`
- note count gain: `32`
- unique density values: `4`
- duration diversity / IOI diversity: `0.0625 / 0.0421`
- qualified: `true`
- flags: ``
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/midi/rank_02_seed_632_dead_air_density_repair.mid`

### Rank 3

- seed: `638`
- source dead-air: `0.6032`
- repaired dead-air: `0.2211`
- dead-air gain: `0.3821`
- note count gain: `32`
- unique density values: `4`
- duration diversity / IOI diversity: `0.0833 / 0.0421`
- qualified: `true`
- flags: ``
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/midi/rank_03_seed_638_dead_air_density_repair.mid`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- reason: `dead-air and density repair target passed; route repaired MIDI to audio package`
- next recommended issue: `Stage B MIDI-to-solo phrase-bank dead-air density repair audio package`

## Not Proven

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `phrase_bank_musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
