#!/bin/zsh
# Generation driver for issue #1832 (docs/experiments/Q8_DENSITY.md). 32 locked sessions of 48 seeds each:
# for each prompt 0-3 and seed chunk 1000-1047, 1048-1095, 1096-1143, 1144-1191, both precisions back to back,
# order alternating per chunk (fp32 first on even chunk index). Each session takes the GPU lock with mkdir, waits
# while thermal pressure is above 0 (recorded), and releases the lock right after. q8_free_gen.py skips pairs
# already in the output file, so the driver can be restarted.
#   q8_density_run.sh <run dir>
set -u
R=$1
mkdir -p "$R"
HERE=${0:A:h}
T3=/Users/ohhalim/git_box/t3_aria/.venv/bin/python
ARIA=/Users/ohhalim/git_box/t3_aria/aria
CKPT=/Users/ohhalim/git_box/t3_aria/ckpt/model-gen.safetensors
DATA=/Users/ohhalim/git_box/mehldau_aria/data.json
LOCK=/Users/ohhalim/git_box/.gpu_lock
thermal() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}' }
i=0
for p in 0 1 2 3; do
  for a in 1000 1048 1096 1144; do
    b=$((a + 47))
    if (( i % 2 == 0 )); then order=(fp32 q8); else order=(q8 fp32); fi
    for prec in $order; do
      until mkdir "$LOCK" 2>/dev/null; do sleep 30; done
      echo "fork #1832 q8-density $prec p$p s$a-$b $(date +%T)" > "$LOCK/owner"
      while [ "$(thermal)" != "0" ]; do echo "thermal $(thermal) wait $(date +%T)" >> "$R/run.log"; sleep 60; done
      echo "start $prec p$p s$a-$b $(date +%T) thermal $(thermal)" >> "$R/run.log"
      "$T3" "$HERE/q8_free_gen.py" --generate "$ARIA" "$CKPT" "$DATA" "$prec" "$R/$prec.jsonl" --prompts $p --seeds $a-$b \
        >> "$R/gen_$prec.log" 2>&1
      echo "end $prec p$p s$a-$b exit $? $(date +%T) thermal $(thermal)" >> "$R/run.log"
      rm -f "$LOCK/owner"
      rmdir "$LOCK"
    done
    i=$((i + 1))
  done
done
echo "done $(date +%T)" >> "$R/run.log"
