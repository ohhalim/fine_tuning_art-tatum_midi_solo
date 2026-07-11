# Stage B MIDI-to-Solo Model-Direct Generation Repair

## Summary

- boundary: `stage_b_midi_to_solo_model_direct_generation_repair`
- next boundary: `stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke`
- technical MVP preserved: `true`
- current checkpoint sequence budget sufficient: `false`
- direct repair required: `true`
- model-direct generation quality claimed: `false`

## Contract

- current generation source: `context_conditioned_fallback`
- required generation source: `model_checkpoint_direct`
- target solo bars: `8`
- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`

## Sequence Budget

- current max sequence: `96`
- overhead tokens: `27`
- minimum contract tokens: `123`
- direct note capacity under current budget: `17`
- even note groups per bar capacity: `2`
- recommended max sequence: `160`

## Repair Scope

- primary blocker: `scale_smoke_sequence_budget`
- current skip reason: `scale-smoke max_sequence 96 supports 17 direct Stage B notes after 8-bar bar/chord overhead, below MVP minimum 24`
- next validation target: `stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke`

## Not Proven

- `model_checkpoint_direct_8bar_generation`
- `model_checkpoint_direct_8bar_generation_quality`
- `human_audio_preference`
- `broad_trained_model_quality`
- `brad_style_adaptation`
