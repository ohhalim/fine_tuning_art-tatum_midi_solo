#!/usr/bin/env bash
# Read the chosen budgets for the 3x3 stage (TATUM_VS_MEHLDAU.md §3).
# Prints "BT BM" and exits 0 when both artists have a budget; otherwise writes
# ADAPTATION_FAILED to <out_dir>/adaptation_failed.txt and exits 2.
# Usage: tvm_check_budgets.sh <budget_file> <out_dir>
set -euo pipefail
file="$1"; out="$2"
BT=""; BM=""
if [ -f "$file" ]; then
  BT=$(awk '$1=="tatum"{print $2}' "$file")
  BM=$(awk '$1=="mehldau"{print $2}' "$file")
fi
if [ -z "$BT" ] || [ -z "$BM" ]; then
  echo "ADAPTATION_FAILED tatum='${BT}' mehldau='${BM}' -> 3x3/generation skipped" | tee "$out/adaptation_failed.txt" >&2
  exit 2
fi
echo "$BT $BM"
