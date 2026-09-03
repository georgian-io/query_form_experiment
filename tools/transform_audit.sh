#!/bin/bash
# §5.6 intent-drift audit for the transform phase: for each (dataset × condition) judge all 4
# generators' rewrites with a cross-lineage judge. The 6 combos run in parallel. Resumable —
# 06_audit caches verdicts per (original,rewrite,judge,rubric), so a reap re-hits cache and only new
# judgments run; a combo's output file is written only when its full sample completes. Under launchd
# (KeepAlive SuccessfulExit=false): exit 1 while any of the 6 audit files is missing, exit 0 when all
# 6 exist.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.." || exit 3
UV="$HOME/.local/bin/uv"

audit_one() {
  local ds="$1" cond="$2"
  local out="results/audit__${ds}__${cond}.json"
  [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] skip $ds $cond"; return; }
  echo "[$(date +%H:%M:%S)] === audit: $ds $cond ==="
  "$UV" run python scripts/06_audit.py --dataset "$ds" --condition "$cond" --sample 120 \
    || echo "[$(date +%H:%M:%S)] FAILED: $ds $cond"
}

for ds in curev1_en trec_covid; do
  for cond in paraphrase terse verbose; do
    audit_one "$ds" "$cond" &
  done
done
wait

have=$(ls results/audit__curev1_en__paraphrase.json results/audit__curev1_en__terse.json \
  results/audit__curev1_en__verbose.json results/audit__trec_covid__paraphrase.json \
  results/audit__trec_covid__terse.json results/audit__trec_covid__verbose.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date +%H:%M:%S)] audit files: ${have}/6"
[ "$have" -ge 6 ] && { echo "[$(date +%H:%M:%S)] === AUDIT COMPLETE ==="; exit 0; }
exit 1
