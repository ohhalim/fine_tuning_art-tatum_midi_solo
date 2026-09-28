#!/usr/bin/env bash
# Tatum vs Mehldau comparison pipeline (docs/experiments/TATUM_VS_MEHLDAU.md), after the common base.
# Usage: PY=<python> MAIN=<main repo> bash scripts/run_tvm_pipeline.sh
set -euo pipefail
PY="${PY:?}"; MAIN="${MAIN:?}"
BASE="outputs/tvm/common_base/checkpoint_epoch8.pt"
ARMB="$MAIN/outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt"
GENERIC="outputs/tatum_diag/generic_probe_list.json"   # 100 probes, Tatum and Mehldau excluded
PRIMER="outputs/chord_ab/ii_V_I.mid"
O="outputs/tvm"; W="$(pwd)"

# P0 gate
"$PY" scripts/compare_base_ce.py --checkpoint armB="$ARMB" --checkpoint common="$BASE" \
  --target-dir data/tvm/tatum16 --generic-list "$GENERIC" --output "$O/p0_gate_tatum.json"
"$PY" scripts/compare_base_ce.py --checkpoint armB="$ARMB" --checkpoint common="$BASE" \
  --target-dir data/tvm/mehldau16 --generic-list "$GENERIC" --output "$O/p0_gate_mehldau.json"

# P1 CV (each fold starts from the common base with a fresh optimizer)
for artist in tatum mehldau; do
  reports=()
  for k in 0 1 2 3; do
    "$PY" scripts/run_mehldau_update_budget_diag.py --device mps --seed 42 --checkpoint "$BASE" \
      --data-dir "data/tvm/cv_$artist/fold$k" --lora-targets out_proj --primer "$PRIMER" \
      --output-dir "$O/cv_${artist}_fold$k" --planned-epochs 128 --budget-seconds 5000 \
      --snapshot-updates 0,16,32,64,128 --gen-seeds 42 --gen-bars 1 > "$O/cv_${artist}_fold$k.log" 2>&1
    "$PY" scripts/eval_mehldau_snapshots.py --device mps --no-generate --updates 0,16,32,64,128 \
      --checkpoint "$BASE" --snapshot-dir "$O/cv_${artist}_fold$k" --data-dir "data/tvm/cv_$artist/fold$k" \
      --target-name "$artist" --validity-json "$GENERIC" --primer "$PRIMER" \
      --output-dir "$O/cv_eval_${artist}_fold$k" > "$O/cv_eval_${artist}_fold$k.log" 2>&1
    reports+=(--fold-report "$O/cv_eval_${artist}_fold$k/report.json")
  done
  "$PY" scripts/aggregate_song_cv.py "${reports[@]}" --require-target-drop --tie-tolerance 0.001 \
    --output "$O/cv_$artist.json" > "$O/cv_$artist.txt"
done
echo P1_DONE

# P2 final adapters on all 16 songs, cosine 128, snapshot at the chosen update
for artist in tatum mehldau; do
  B=$("$PY" -c "import json;c=json.load(open('$O/cv_$artist.json'))['chosen'];print(c['update'] if c else '')")
  if [ -z "$B" ]; then echo "$artist: no eligible budget (adaptation failed)"; continue; fi
  U=$(printf "%03d" "$B")
  "$PY" scripts/run_mehldau_update_budget_diag.py --device mps --seed 42 --checkpoint "$BASE" \
    --data-dir "data/tvm/${artist}16" --lora-targets out_proj --primer "$PRIMER" \
    --output-dir "$O/final_$artist" --planned-epochs 128 --budget-seconds 5000 \
    --snapshot-updates "0,$B" --gen-seeds 42 --gen-bars 1 > "$O/final_$artist.log" 2>&1
  "$PY" scripts/eval_mehldau_snapshots.py --device mps --updates "0,$B" --seeds 1,2,3,4 --gen-tokens 768 \
    --checkpoint "$BASE" --snapshot-dir "$O/final_$artist" --data-dir "data/tvm/${artist}16" \
    --target-name "$artist" --validity-json "$GENERIC" --primer "$PRIMER" \
    --output-dir "$O/final_eval_$artist" > "$O/final_eval_$artist.log" 2>&1
  FORCE_CPU=1 "$PY" scripts/export_lora_snapshot.py --base "$BASE" --snapshot "$O/final_$artist/lora_update$U.pt" \
    --output "$O/export_$artist/checkpoint_update$B.pt" \
    --note "common base (Tatum+Mehldau excluded) + out_proj, $artist train16, $B updates" > "$O/export_$artist.log" 2>&1
  echo "$artist $B" >> "$O/chosen_budgets.txt"
done
echo P2_DONE

# P3 3x3
BT=$(awk '$1=="tatum"{print $2}' "$O/chosen_budgets.txt"); BM=$(awk '$1=="mehldau"{print $2}' "$O/chosen_budgets.txt")
if [ -z "$BT" ] || [ -z "$BM" ]; then
  # A pre-registered adaptation failure: report it instead of reading a checkpoint that was never made.
  echo "ADAPTATION_FAILED tatum='${BT}' mehldau='${BM}' -> 3x3/generation skipped" | tee "$O/adaptation_failed.txt"
  exit 2
fi
"$PY" scripts/eval_cross_artist.py --model base="$BASE" \
  --model tatum_adapter="$O/export_tatum/checkpoint_update$BT.pt" \
  --model mehldau_adapter="$O/export_mehldau/checkpoint_update$BM.pt" \
  --column tatum_fresh12=data/tvm/holdout_tatum_fresh12 --column mehldau_val2=data/tvm/holdout_mehldau_val2 \
  --column tatum_val12=data/tvm/holdout_tatum_val12 --generic-list "$GENERIC" --output "$O/p3_cross.json" > "$O/p3_cross.txt"
echo P3_DONE

# P4 generation descriptors + runtime
"$PY" scripts/describe_generations.py \
  --model base="$O/final_eval_tatum/generated_tokens.json:0" \
  --model tatum_adapter="$O/final_eval_tatum/generated_tokens.json:$BT" \
  --model mehldau_adapter="$O/final_eval_mehldau/generated_tokens.json:$BM" \
  --train-set tatum16=data/tvm/tatum16/train --train-set mehldau16=data/tvm/mehldau16/train \
  --output "$O/p4_generation.json" > "$O/p4_generation.txt"
for m in "base:$W/$BASE" "tatum_adapter:$W/$O/export_tatum/checkpoint_update$BT.pt" "mehldau_adapter:$W/$O/export_mehldau/checkpoint_update$BM.pt"; do
  n=${m%%:*}; c=${m#*:}
  FORCE_CPU=1 "$PY" scripts/run_continuous_jazz.py --checkpoint "$c" --conditioning-midi "$PRIMER" \
    --bars 8 --bpm 128 --seed 42 --chords Dm7,G7,Cmaj7,A7 --capture --chord-primer --chord-blocks-per-bar 2 \
    --output-dir "$O/runtime_$n" > "$O/runtime_$n.log" 2>&1
done
echo PIPELINE_DONE
