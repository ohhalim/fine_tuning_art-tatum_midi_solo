#!/usr/bin/env bash
# Post-training steps for docs/experiments/FINAL_TATUM.md
set -euo pipefail
PY="${PY:?}"; W="$(pwd)"
BASE="outputs/tvm/common_base/checkpoint_epoch8.pt"
GENERIC="outputs/tatum_diag/generic_probe_list.json"
PRIMER="outputs/chord_ab/ii_V_I.mid"
O="outputs/final_tatum"
"$PY" scripts/eval_mehldau_snapshots.py --device mps --no-generate --checkpoint "$BASE" \
  --snapshot-dir "$O/run" --data-dir data/tvm/tatum98 --target-name tatum --validity-json "$GENERIC" \
  --primer "$PRIMER" --output-dir "$O/select" > "$O/select.log" 2>&1
B=$("$PY" scripts/select_by_val.py "$O/select/report.json")
[ -n "$B" ] || { echo "NO_ELIGIBLE_SNAPSHOT" | tee "$O/failed.txt"; exit 2; }
echo "$B" > "$O/chosen.txt"; U=$(printf "%03d" "$B")
FORCE_CPU=1 "$PY" scripts/export_lora_snapshot.py --base "$BASE" --snapshot "$O/run/lora_update$U.pt" \
  --output "$O/export/checkpoint_update$B.pt" --note "FINAL Tatum: common base + out_proj+qkv, 98 songs, $B updates (val12-selected)" > "$O/export.log" 2>&1
"$PY" scripts/eval_cross_artist.py --model base="$BASE" \
  --model tatum16="outputs/tvm/export_tatum/checkpoint_update128.pt" \
  --model tatum_final="$O/export/checkpoint_update$B.pt" \
  --column tatum_fresh12=data/tvm/holdout_tatum_fresh12 --column mehldau16=data/tvm/mehldau16/train \
  --column mehldau_val2=data/tvm/holdout_mehldau_val2 --generic-list "$GENERIC" --output "$O/cross.json" > "$O/cross.txt" 2>&1
"$PY" scripts/eval_mehldau_snapshots.py --device mps --updates "0,$B" --seeds 1,2,3,4 --gen-tokens 768 \
  --checkpoint "$BASE" --snapshot-dir "$O/run" --data-dir data/tvm/tatum98 --target-name tatum \
  --validity-json "$GENERIC" --primer "$PRIMER" --output-dir "$O/gen" > "$O/gen.log" 2>&1
"$PY" scripts/describe_generations.py --model base="$O/gen/generated_tokens.json:0" \
  --model tatum_final="$O/gen/generated_tokens.json:$B" --train-set tatum98=data/tvm/tatum98/train \
  --output "$O/gen_descriptors.json" > "$O/gen_descriptors.txt" 2>&1
"$PY" scripts/run_tempo_sweep.py --python "$PY" --bpms 128 --bars 16 --seeds 42,43,44 \
  --extra "--chord-primer --chord-blocks-per-bar 2 --chords Dm7,G7,Cmaj7,A7" \
  --model tatum_final="$W/$O/export/checkpoint_update$B.pt" --primer "$W/$PRIMER" --output-dir "$O/runtime16" > "$O/runtime16.log" 2>&1
echo FINAL_TATUM_POST_DONE
