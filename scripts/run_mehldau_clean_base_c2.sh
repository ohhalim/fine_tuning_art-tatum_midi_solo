#!/usr/bin/env bash
# C2 final Mehldau adapter on the Mehldau-free base (docs/experiments/MEHLDAU_CLEAN_BASE.md).
# Usage: PY=<python> MAIN=<main repo> BUDGET=<update> bash scripts/run_mehldau_clean_base_c2.sh
set -euo pipefail
PY="${PY:?}"; MAIN="${MAIN:?}"; BUDGET="${BUDGET:?}"
BASE_NEW="outputs/clean_base/nomehldau_full/checkpoint_epoch8.pt"
GENERIC="outputs/mehldau_diag/style_validity_v1.json"
PRIMER="outputs/chord_ab/ii_V_I.mid"
OUT="outputs/clean_base"
W="$(pwd)"
U=$(printf "%03d" "$BUDGET")

"$PY" scripts/run_mehldau_update_budget_diag.py --device mps --seed 42 --checkpoint "$BASE_NEW" \
  --data-dir "$MAIN/data/mehldau_full" --lora-targets out_proj --primer "$PRIMER" \
  --output-dir "$OUT/c2_final" --planned-epochs 128 --budget-seconds 5000 \
  --snapshot-updates "0,$BUDGET" --gen-seeds 42 --gen-bars 1 > "$OUT/c2_final.log" 2>&1
"$PY" scripts/eval_mehldau_snapshots.py --device mps --updates "0,$BUDGET" --seeds 1,2,3,4 \
  --gen-tokens 768 --checkpoint "$BASE_NEW" --snapshot-dir "$OUT/c2_final" \
  --data-dir "$MAIN/data/mehldau_full" --target-name mehldau --validity-json "$GENERIC" \
  --primer "$PRIMER" --output-dir "$OUT/c2_eval" > "$OUT/c2_eval.log" 2>&1
FORCE_CPU=1 "$PY" scripts/export_lora_snapshot.py --base "$BASE_NEW" \
  --snapshot "$OUT/c2_final/lora_update$U.pt" \
  --output "$OUT/c2_export/checkpoint_update$BUDGET.pt" \
  --note "Mehldau-free base + out_proj LoRA, 16 Mehldau songs, $BUDGET updates" > "$OUT/c2_export.log" 2>&1
for m in "clean_base:$W/$BASE_NEW" "clean_mehldau:$W/$OUT/c2_export/checkpoint_update$BUDGET.pt"; do
  n=${m%%:*}; c=${m#*:}
  FORCE_CPU=1 "$PY" scripts/run_continuous_jazz.py --checkpoint "$c" --conditioning-midi "$PRIMER" \
    --bars 8 --bpm 128 --capture --chord-primer --chord-blocks-per-bar 2 \
    --output-dir "$OUT/c2_runtime_$n" > "$OUT/c2_runtime_$n.log" 2>&1
done
echo C2_DONE
