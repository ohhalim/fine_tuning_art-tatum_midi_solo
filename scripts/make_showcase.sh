#!/usr/bin/env bash
# Showcase MIDI for the personalised solo models (docs/PERSONALIZATION_STATUS.md).
# Same primer, chords, seed and length for every preset of
# scripts/play_personalized.py: base (no artist adapter), tatum, mehldau, and
# swap (Tatum <-> Mehldau every 4 bars in one session). Played through the
# normal runtime on a virtual port and captured.
# Descriptive material only: no quality or style claim, nobody has listened.
# Usage: PY=<python> OUT=<dir> bash scripts/make_showcase.sh
set -euo pipefail
PY="${PY:?set PY to the project python}"
OUT="${OUT:?set OUT to an output directory}"
SEED="${SEED:-42}"
NAMES=(ii_V_I_C blues_F minor_ii_V_i_C)
CHORDS=("Dm7,G7,Cmaj7,Cmaj7" "F7,Bb7,F7,F7,Bb7,Bb7,F7,F7,Gm7,C7,F7,C7" "Dm7b5,G7,Cm7,Cm7")
BARS=(16 12 16)
BPMS=(128 120 128)
mkdir -p "$OUT/midi" "$OUT/runs"
export FORCE_CPU=1
for i in 0 1 2; do
  for preset in base tatum mehldau swap; do
    d="$OUT/runs/${NAMES[$i]}_$preset"
    "$PY" scripts/play_personalized.py --preset "$preset" --chords "${CHORDS[$i]}" \
      --bars "${BARS[$i]}" --bpm "${BPMS[$i]}" --seed "$SEED" --capture --output-dir "$d" > "$d.log" 2>&1
    cp "$d/played.mid" "$OUT/midi/${NAMES[$i]}_${preset}.mid"
  done
done
echo SHOWCASE_DONE
