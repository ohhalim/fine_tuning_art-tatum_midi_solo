#!/usr/bin/env bash
# C0 gate + C1 song-level CV for docs/experiments/MEHLDAU_CLEAN_BASE.md.
# Usage: PY=<python> MAIN=<main repo> bash scripts/run_mehldau_clean_base_c1.sh
set -euo pipefail
PY="${PY:?python}"; MAIN="${MAIN:?main repo}"
BASE_NEW="outputs/clean_base/nomehldau_full/checkpoint_epoch8.pt"
BASE_OLD="$MAIN/outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt"
GENERIC="outputs/mehldau_diag/style_validity_v1.json"
PRIMER="outputs/chord_ab/ii_V_I.mid"
OUT="outputs/clean_base"

# C0 gate: same crops, both bases
"$PY" scripts/compare_base_ce.py --checkpoint armB="$BASE_OLD" --checkpoint clean="$BASE_NEW" \
  --target-dir "$MAIN/data/mehldau_full" --generic-list "$GENERIC" --output "$OUT/c0_gate.json"

# C1: 4 folds, out_proj, cosine over 128 updates (12 songs / batch 4 / accumulation 4 = 1 update/epoch)
for k in 0 1 2 3; do
  "$PY" scripts/run_mehldau_update_budget_diag.py --device mps --seed 42 \
    --checkpoint "$BASE_NEW" --data-dir "data/mehldau_cv/fold$k" --lora-targets out_proj \
    --primer "$PRIMER" --output-dir "$OUT/c1_fold$k" --planned-epochs 128 --budget-seconds 5000 \
    --snapshot-updates 0,16,32,64,128 --gen-seeds 42 --gen-bars 1 > "$OUT/c1_fold$k.log" 2>&1
  "$PY" scripts/eval_mehldau_snapshots.py --device mps --no-generate --updates 0,16,32,64,128 \
    --checkpoint "$BASE_NEW" --snapshot-dir "$OUT/c1_fold$k" --data-dir "data/mehldau_cv/fold$k" \
    --target-name mehldau --validity-json "$GENERIC" --primer "$PRIMER" \
    --output-dir "$OUT/c1_eval_fold$k" > "$OUT/c1_eval_fold$k.log" 2>&1
done
"$PY" scripts/aggregate_song_cv.py \
  --fold-report "$OUT/c1_eval_fold0/report.json" --fold-report "$OUT/c1_eval_fold1/report.json" \
  --fold-report "$OUT/c1_eval_fold2/report.json" --fold-report "$OUT/c1_eval_fold3/report.json" \
  --output "$OUT/c1_cv.json"
echo C1_DONE
