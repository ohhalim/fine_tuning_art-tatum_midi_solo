# Stage B MIDI-to-Solo Post-MVP Quality Iteration Plan Source Context Refresh

## Summary

- boundary: `stage_b_midi_to_solo_post_mvp_quality_iteration_plan`
- source boundary: `stage_b_midi_to_solo_final_status_audit`
- next boundary: `stage_b_midi_to_solo_quality_rubric_baseline`
- selected target: `quality_rubric_baseline`
- technical MVP complete: `true`
- local review ready: `true`
- outside-soloing repair evidence ready: `true`
- outside-soloing repair WAV count: `6`
- outside-soloing source objective pitch-role risk: `5`
- outside-soloing source pitch-role risk before / after / delta: `5` / `2` / `3`
- outside-soloing source repair targeted: `false`
- outside-soloing source residual risk preserved: `true`
- outside-soloing current repair pitch-role risk after / delta: `0` / `2`

## Required Work

- `quality_rubric_baseline`: define MIDI-evidence quality rubric before another repair sweep
- `candidate_failure_labeling`: label current candidates against sparse-output, songlike-melody, outside-soloing, monotony, and phrase-shape failures
- `targeted_quality_repair_sweep`: run repair/generation sweep against the highest-count failure labels
- `audio_review_package`: render selected repaired candidates for listening comparison

## Claim Boundary

- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`
- broad trained model quality claimed: `false`

## Next

- `Stage B MIDI-to-solo quality rubric baseline source-context refresh`
