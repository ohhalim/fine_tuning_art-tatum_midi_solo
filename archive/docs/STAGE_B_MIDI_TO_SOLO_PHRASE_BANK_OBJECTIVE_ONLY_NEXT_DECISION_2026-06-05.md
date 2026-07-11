# Stage B MIDI-to-Solo Phrase-Bank Objective-Only Next Decision

## Summary

- boundary: `stage_b_midi_to_solo_phrase_bank_objective_only_next_decision`
- next boundary: `stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe`
- review basis: `objective_midi_and_wav_metadata_only`
- candidate count: `3`
- objective keep candidate count: `0`
- repair required candidate count: `3`
- all candidates require repair: `true`
- dead-air range: `0.5873 - 0.6032`
- preference fill allowed: `false`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

## Candidate Objective Review

### Rank 1

- seed: `635`
- notes / unique pitches / max simultaneous: `64 / 22 / 1`
- dead-air / phrase coverage: `0.5873 / 1.0000`
- duration diversity / IOI diversity: `0.0938 / 0.0952`
- approach resolution / repeated pitch ratio: `0.3529 / 0.6562`
- repair required: `true`
- risk flags: `dead_air_ratio_above_review_threshold, uniform_bar_note_density, low_duration_diversity, low_ioi_diversity, low_approach_resolution, high_pitch_reuse_ratio, no_leap_motion`

### Rank 2

- seed: `632`
- notes / unique pitches / max simultaneous: `64 / 21 / 1`
- dead-air / phrase coverage: `0.5873 / 1.0000`
- duration diversity / IOI diversity: `0.0938 / 0.0952`
- approach resolution / repeated pitch ratio: `0.3714 / 0.6719`
- repair required: `true`
- risk flags: `dead_air_ratio_above_review_threshold, uniform_bar_note_density, low_duration_diversity, low_ioi_diversity, low_approach_resolution, high_pitch_reuse_ratio, no_leap_motion`

### Rank 3

- seed: `638`
- notes / unique pitches / max simultaneous: `64 / 22 / 1`
- dead-air / phrase coverage: `0.6032 / 1.0000`
- duration diversity / IOI diversity: `0.0781 / 0.0952`
- approach resolution / repeated pitch ratio: `0.3056 / 0.6562`
- repair required: `true`
- risk flags: `dead_air_ratio_above_review_threshold, uniform_bar_note_density, low_duration_diversity, low_ioi_diversity, low_approach_resolution, high_pitch_reuse_ratio, no_leap_motion`

## Decision

- auto progress allowed: `true`
- critical user input required: `false`
- reason: `phrase-bank candidates require objective repair before CLI MVP packaging`
- next recommended issue: `Stage B MIDI-to-solo phrase-bank dead-air density repair probe`

## Not Proven

- `human_audio_preference`
- `midi_to_solo_musical_quality`
- `phrase_bank_musical_quality`
- `broad_trained_model_quality`
- `brad_style_adaptation`
