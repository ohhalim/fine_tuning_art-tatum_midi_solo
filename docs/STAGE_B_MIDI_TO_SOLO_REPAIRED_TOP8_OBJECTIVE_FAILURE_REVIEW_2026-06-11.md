# Music Transformer Solo Yield Repaired Top8 Objective Failure Review

## Summary

- candidate count: `8`
- failed candidate count: `8`
- final landing not chord tone: `8`
- package low chord-tone ratio: `8`
- MIDI low chord-tone ratio: `6`
- dead-air still high: `3`
- selected next target: `chord_tone_landing_repair`
- next boundary: `music_transformer_solo_yield_chord_tone_landing_repair_sweep`
- musical quality claimed: `false`

## Failure Label Counts

- `dead_air_still_high`: `3`
- `final_landing_not_chord_tone`: `8`
- `low_note_count_for_4bar`: `2`
- `midi_low_chord_tone_ratio`: `6`
- `package_low_chord_tone_ratio`: `8`
- `weak_direction_change`: `2`
- `wide_interval_review`: `2`

## Candidates

- candidate `1` / `minor_backdoor`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`
  - package dead-air: `0.5938`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4848`
  - final landing: `65` over `Ebmaj7`, chord-tone `false`
- candidate `2` / `minor_backdoor`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`, `weak_direction_change`, `wide_interval_review`
  - package dead-air: `0.5152`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4412`
  - final landing: `72` over `Ebmaj7`, chord-tone `false`
- candidate `3` / `major_ii_v_turnaround`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`, `wide_interval_review`
  - package dead-air: `0.6452`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4375`
  - final landing: `77` over `A7`, chord-tone `false`
- candidate `4` / `major_ii_v_turnaround`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`
  - package dead-air: `0.5484`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4688`
  - final landing: `71` over `A7`, chord-tone `false`
- candidate `5` / `dominant_cycle`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `dead_air_still_high`
  - package dead-air: `0.7241`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.5000`
  - final landing: `72` over `G7`, chord-tone `false`
- candidate `6` / `dominant_cycle`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `dead_air_still_high`, `low_note_count_for_4bar`
  - package dead-air: `0.6667`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.5714`
  - final landing: `75` over `G7`, chord-tone `false`
- candidate `7` / `rhythm_turnaround`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`, `dead_air_still_high`, `low_note_count_for_4bar`
  - package dead-air: `0.7143`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4483`
  - final landing: `76` over `F7`, chord-tone `false`
- candidate `8` / `rhythm_turnaround`
  - labels: `final_landing_not_chord_tone`, `package_low_chord_tone_ratio`, `midi_low_chord_tone_ratio`, `weak_direction_change`
  - package dead-air: `0.6000`
  - package chord-tone ratio: `0.4444`
  - MIDI chord-tone ratio: `0.4839`
  - final landing: `70` over `F7`, chord-tone `false`

## Not Proven

- `human_audio_preference`
- `stable_jazz_solo_quality`
- `artist_level_long_solo_generation`
- `production_ready_improviser`
