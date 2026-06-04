# Stage B MIDI-to-Solo Controlled Scale Checkpoint Repair Decision

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe`
- selected target: `target_density_collapse_postprocess_repair`
- postprocess-only repair selected: `false`
- audio review selected: `false`
- training scale change selected: `false`
- MIDI-to-solo musical quality claimed: `false`

## Evidence

- train / val records: `512` / `128`
- best validation loss: `5.1061`
- sample count: `3`
- valid / strict / grammar: `0` / `0` / `3`
- note count failure count: `3`
- collapse warning sample count / rate: `3` / `1.0`
- avg onset / sustained coverage ratio: `0.08333333333333333` / `0.16666666666666666`
- avg / max postprocess removal ratio: `0.809042809042809` / `0.8636363636363636`

## Repair Targets

- `increase_note_density_before_postprocess`
- `reduce_postprocess_removed_majority`
- `preserve_grammar_gate`
- `improve_onset_sustained_coverage`
- `track_repeated_position_pitch_pair`
- `preserve_no_quality_claim`

## Failure Reasons

- `note count too low: 4 < 6; collapse=postprocess_removed_majority`: `1`
- `note count too low: 3 < 6; collapse=repeated_position_pitch,postprocess_removed_majority`: `1`
- `note count too low: 3 < 6; collapse=postprocess_removed_majority`: `1`

## Not Proven

- `repair_probe_result`
- `quality_root_cause`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
