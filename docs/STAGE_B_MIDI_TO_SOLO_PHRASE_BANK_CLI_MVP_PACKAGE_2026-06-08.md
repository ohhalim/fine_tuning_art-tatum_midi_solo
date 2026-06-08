# Stage B MIDI-to-Solo Phrase-Bank CLI MVP Package

## Summary

- boundary: `stage_b_midi_to_solo_phrase_bank_cli_mvp_package`
- next boundary: `stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke`
- input MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package/input/fixture.mid`
- CLI MVP package completed: `true`
- ranked repaired MIDI exported: `true`
- candidate count: `3`
- objective supported candidate count: `3`
- dead-air range: `0.1895 - 0.2211`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Command

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py --input_midi outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package/input/fixture.mid --run_id harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package
```

## Candidate Manifest

### Rank 1

- seed: `635`
- objective supported: `true`
- notes / unique pitches / max simultaneous: `96 / 23 / 1`
- dead-air / phrase coverage: `0.1895 / 1.0000`
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package/midi/rank_01_seed_635_dead_air_density_repair.mid`

### Rank 2

- seed: `632`
- objective supported: `true`
- notes / unique pitches / max simultaneous: `96 / 21 / 1`
- dead-air / phrase coverage: `0.2105 / 1.0000`
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package/midi/rank_02_seed_632_dead_air_density_repair.mid`

### Rank 3

- seed: `638`
- objective supported: `true`
- notes / unique pitches / max simultaneous: `96 / 22 / 1`
- dead-air / phrase coverage: `0.2211 / 1.0000`
- repaired MIDI: `outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package/midi/rank_03_seed_638_dead_air_density_repair.mid`

## Claim Boundary

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `phrase_bank_musical_quality`
- `audio_rendered_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
- `production_ready`

## Next

- `Stage B MIDI-to-solo phrase-bank CLI user-input smoke`
