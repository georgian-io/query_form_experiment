#!/bin/bash
# Stage 1 of the transform phase: generate paraphrase/terse/verbose rewrites for both new datasets
# across all 4 generator lineages. The 4 generators are independent providers (Vertex / Anthropic /
# OpenAI / Vertex-MaaS), so they run as PARALLEL lanes; within a lane the (dataset,condition) combos
# run sequentially. Resumable — 02_run_transform caches per (dataset,qid,condition,generator,
# prompt_version), so a reap re-hits cache and only new queries generate. Under launchd
# (KeepAlive SuccessfulExit=false): exits 1 while any of the 24 sets is missing (launchd resumes),
# exits 0 only when all 24 exist.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.." || exit 3
UV="$HOME/.local/bin/uv"

GENS=("gemini-2.5-pro" "claude-opus-4-8" "gpt-5.1-2025-11-13" "meta-llama/llama-3.3-70b-instruct")
CONDS=("paraphrase" "terse" "verbose")
DATASETS=("curev1_en" "trec_covid")
flat() { echo "$1" | sed 's#/#__#g'; }

gen_lane() {
  local gen="$1"
  for ds in "${DATASETS[@]}"; do
    for cond in "${CONDS[@]}"; do
      local out="results/queries__${ds}__${cond}__$(flat "$gen").json"
      [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] skip $ds $cond $gen"; continue; }
      echo "[$(date +%H:%M:%S)] === generate: $ds $cond $gen ==="
      "$UV" run python scripts/02_run_transform.py --dataset "$ds" --condition "$cond" --generator "$gen" \
        || echo "[$(date +%H:%M:%S)] FAILED: $ds $cond $gen"
    done
  done
  echo "[$(date +%H:%M:%S)] lane done: $gen"
}

for gen in "${GENS[@]}"; do gen_lane "$gen" & done
wait

have=$(ls results/queries__curev1_en__paraphrase__*.json results/queries__curev1_en__terse__*.json \
  results/queries__curev1_en__verbose__*.json results/queries__trec_covid__paraphrase__*.json \
  results/queries__trec_covid__terse__*.json results/queries__trec_covid__verbose__*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date +%H:%M:%S)] sets complete: ${have}/24"
[ "$have" -ge 24 ] && { echo "[$(date +%H:%M:%S)] === GENERATION COMPLETE ==="; exit 0; }
exit 1
