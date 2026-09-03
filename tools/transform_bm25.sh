#!/bin/bash
# BM25 lexical control for the transform conditions — the pilot's mechanism decomposition. One
# lexical run per (dataset × condition × generator) against a fixed reference namespace
# (gemini-embedding-001; all model namespaces share the same BM25 text). No embedding, so it only
# touches turbopuffer — safe to run concurrent with the audit. Parallel by generator. Resumable;
# under launchd: exit 1 while any of 24 files missing, exit 0 when all present.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd "$(dirname "$0")/.." || exit 3
UV="$HOME/.local/bin/uv"
REF="gemini-embedding-001"
GENS=("gemini-2.5-pro" "claude-opus-4-8" "gpt-5.1-2025-11-13" "meta-llama/llama-3.3-70b-instruct")
flat() { echo "$1" | sed 's#/#__#g'; }
lane() {
  local gen="$1"
  for ds in curev1_en trec_covid; do for cond in paraphrase terse verbose; do
    out="results/bm25__${ds}__${REF}__${cond}__$(flat "$gen").json"
    [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] skip $ds $cond $gen"; continue; }
    echo "[$(date +%H:%M:%S)] === bm25 $ds $cond $gen ==="
    "$UV" run python scripts/03_run_retrieval.py --dataset "$ds" --model "$REF" --mode bm25 --condition "$cond" --generator "$gen" \
      || echo "[$(date +%H:%M:%S)] FAILED $ds $cond $gen"
  done; done
}
for g in "${GENS[@]}"; do lane "$g" & done
wait
have=$(ls results/bm25__curev1_en__${REF}__*.json results/bm25__trec_covid__${REF}__*.json 2>/dev/null | wc -l | tr -d ' ')
echo "[$(date +%H:%M:%S)] bm25 files: ${have}/24"
[ "$have" -ge 24 ] && { echo "[$(date +%H:%M:%S)] === BM25 COMPLETE ==="; exit 0; }
exit 1
