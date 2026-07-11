# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Repair Decision

## Summary

- boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision`
- next boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe`
- selected target: `target_density_grammar_collapse_postprocess_repair`
- postprocess-only repair selected: `false`
- audio review selected: `false`
- additional training scale selected: `false`
- MIDI-to-solo musical quality claimed: `false`

## Evidence

- train / val records: `2048` / `512`
- best validation loss: `3.0396`
- sample count: `3`
- valid / strict / grammar: `0` / `0` / `2`
- note count failure count: `3`
- grammar failure count: `1`
- collapse warning sample count / rate: `3` / `1.0`
- avg onset / sustained coverage ratio: `0.11458333333333333` / `0.14583333333333334`
- avg / max postprocess removal ratio: `0.790909090909091` / `0.8`

## Repair Targets

- `increase_note_density_before_postprocess`
- `repair_partial_generation_grammar_loss`
- `reduce_postprocess_removed_majority`
- `improve_onset_sustained_coverage`
- `preserve_selected_scale_checkpoint_evidence`
- `preserve_no_quality_claim`

## Failure Reasons

- `note count too low: 4 < 6; collapse=postprocess_removed_majority`: `1`
- `note count too low: 5 < 6; collapse=postprocess_removed_majority`: `2`

## Strict Failure Reasons

- `midi_review_gate_failed: note count too low: 4 < 6`: `1`
- `postprocess removal ratio too high: 0.800 > 0.490`: `2`
- `midi_review_gate_failed: note count too low: 5 < 6`: `2`
- `grammar_gate_failed`: `1`
- `postprocess removal ratio too high: 0.773 > 0.490`: `1`

## Not Proven

- `repair_probe_result`
- `quality_root_cause`
- `midi_to_solo_musical_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
