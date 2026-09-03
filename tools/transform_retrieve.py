"""Stage 2 of the transform phase: dense retrieval for every transformed condition.

For each (dataset x condition x generator x embedding-model) it runs 03_run_retrieval, which embeds
the transformed query set (short queries — no OOM risk) against the already-built corpus index and
writes results/dense__<ds>__<model>__<cond>__<gen>.json. Resumable: skips cells whose result
exists. Parallel by embedding provider (independent rate limits / the local GPU), like the board
driver. Exit 0 only when every cell has a result; exit 1 otherwise so launchd resumes.
"""
from __future__ import annotations
import json, subprocess, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
_PY = sys.executable
_LOCK = threading.Lock()

DATASETS = ["curev1_en", "trec_covid"]
CONDS = ["paraphrase", "terse", "verbose"]
GENS = ["gemini-2.5-pro", "claude-opus-4-8", "gpt-5.1-2025-11-13", "meta-llama/llama-3.3-70b-instruct"]

# lanes by embedding provider (nemotron excluded — unservable). local split so a few small models
# embed queries concurrently on the GPU.
LANES = {
    "vertex": ["gemini-embedding-001", "text-embedding-004"],
    "voyage": ["voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-3-large", "voyage-law-2",
               "voyage-3.5-int8-512"],
    "openai": ["text-embedding-3-large", "text-embedding-3-small"],
    "cohere": ["embed-v4.0", "embed-multilingual-v3.0"],
    "novita": ["qwen/qwen3-embedding-8b"],
    "local_a": ["BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "BAAI/bge-m3", "BAAI/bge-small-en-v1.5"],
    "local_b": ["sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"],
}
MODELS = [m for v in LANES.values() for m in v]


def flat(x: str) -> str:
    return x.replace("/", "__")


def result_path(ds: str, model: str, cond: str, gen: str) -> Path:
    return RESULTS / f"dense__{ds}__{flat(model)}__{cond}__{flat(gen)}.json"


def say(msg: str) -> None:
    with _LOCK:
        print(msg, flush=True)


def do_cell(ds: str, model: str, cond: str, gen: str) -> None:
    rp = result_path(ds, model, cond, gen)
    if rp.exists():
        return
    p = subprocess.run(
        [_PY, "scripts/03_run_retrieval.py", "--dataset", ds, "--model", model,
         "--mode", "dense", "--condition", cond, "--generator", gen],
        cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0 or not rp.exists():
        say(f"FAIL {ds} {model} {cond} {gen}: {p.stderr.strip()[-200:]}")
    else:
        say(f"ok {ds} {model} {cond} {gen}")


def run_lane(lane: str, models: list[str]) -> None:
    for model in models:
        for ds in DATASETS:
            for cond in CONDS:
                for gen in GENS:
                    do_cell(ds, model, cond, gen)
    say(f"lane {lane}: pass done")


def main() -> None:
    total = len(MODELS) * len(DATASETS) * len(CONDS) * len(GENS)
    have = sum(result_path(ds, m, c, g).exists()
               for m in MODELS for ds in DATASETS for c in CONDS for g in GENS)
    say(f"transform retrieval: {have}/{total} cells done, {len(LANES)} provider lanes")
    threads = [threading.Thread(target=run_lane, args=(ln, ms), name=ln) for ln, ms in LANES.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    have = sum(result_path(ds, m, c, g).exists()
               for m in MODELS for ds in DATASETS for c in CONDS for g in GENS)
    say(f"=== pass complete: {have}/{total} cells ===")
    sys.exit(0 if have >= total else 1)


if __name__ == "__main__":
    main()
