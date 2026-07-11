# Stage B MIDI-to-Solo Context Extraction MVP

## Summary

- boundary: `stage_b_midi_to_solo_context_extraction_mvp`
- next boundary: `stage_b_midi_to_solo_training_resource_probe`
- input MIDI: `outputs/stage_b_midi_to_solo_context_extraction/harness_stage_b_midi_to_solo_context_extraction/fixture.mid`
- context extraction completed: `true`
- MIDI-to-solo MVP claimed: `false`
- harmony analysis quality claimed: `false`

## Context Summary

- tempo BPM: `120.0`
- context bars: `8`
- positions per bar: `16`
- context event count: `128`
- explicit / inferred / carried / unknown chord bars: `0` / `4` / `4` / `0`
- low-confidence bars: `4`
- bass-note bars: `4`

## Bar Contexts

- bar `0`: `C` `maj7`, bass `36`, source `pitch_class_inference`, confidence `0.9`
- bar `1`: `F` `dom7`, bass `41`, source `pitch_class_inference`, confidence `0.9`
- bar `2`: `G` `dom7`, bass `43`, source `pitch_class_inference`, confidence `0.9`
- bar `3`: `C` `maj7`, bass `36`, source `pitch_class_inference`, confidence `0.9`
- bar `4`: `C` `maj7`, bass `None`, source `carry_forward_empty_bar`, confidence `0.45`
- bar `5`: `C` `maj7`, bass `None`, source `carry_forward_empty_bar`, confidence `0.45`
- bar `6`: `C` `maj7`, bass `None`, source `carry_forward_empty_bar`, confidence `0.45`
- bar `7`: `C` `maj7`, bass `None`, source `carry_forward_empty_bar`, confidence `0.45`
