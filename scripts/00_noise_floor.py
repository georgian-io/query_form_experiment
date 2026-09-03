"""Measure the resolution of the instrument, before trusting any ΔNDCG (§5.5 prerequisite).

Everything Phase 2 reports is a difference of two NDCG numbers. If re-embedding the *same*
unchanged queries can move NDCG by X, then any transform effect smaller than X is
indistinguishable from serving variance, and reporting it would be reporting noise.

Two floors, and they are different things:

  STRUCTURAL -- NDCG@10 is not continuous. With n queries, the smallest possible non-zero change
  is one query moving one rank. Nothing below this is expressible, regardless of noise.

  EMPIRICAL -- re-embed the identical query set several times and see how far the score moves.
  Documents are held fixed (indexed once), which matches the real pipeline and isolates the
  query side, the only side Phase 2 varies.

Expect these to differ sharply by route: locally-run pinned weights should be exactly
reproducible, while hosted endpoints have shown ~1e-3 per-element variation from bf16-class
serving nondeterminism.

    uv run python scripts/00_noise_floor.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_datasets, load_experiment  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.index.embed import CachedEmbedder, get_embedder  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# One model per route, so the floor is characterised per provider rather than per model.
PROBES = [
    "sentence-transformers/all-MiniLM-L6-v2",  # local, revision-pinned
    "gemini-embedding-001",                    # vertex, versioned
    "voyage-4-large",                          # voyage
    "text-embedding-3-small",                  # openai
    "embed-v4.0",                              # cohere
    "qwen/qwen3-embedding-8b",                 # novita
]


def structural_floor(qrels, n_queries: int) -> float:
    """Smallest expressible non-zero ΔNDCG@10: one query, one rank.

    For a query with a single relevant document that is a move from rank 1 to rank 2. With
    several relevant documents the smallest step is smaller, so this is computed from the
    dataset's actual rels-per-query distribution rather than assumed.
    """
    step = 1.0 - 1.0 / math.log2(3)                       # rank 1 -> 2, single-gold case
    rels = [len(v) for v in qrels.values()]
    if max(rels) > 1:
        # With r golds, moving the lowest-ranked one by one position changes DCG by less.
        r = max(rels)
        step = (1 / math.log2(r + 1)) - (1 / math.log2(r + 2))
    return step / n_queries


def run(model_id: str, dataset_name: str, repeats: int) -> dict:
    model = load_experiment().model(model_id)
    dc = load_datasets().dataset(dataset_name)
    ds = load_dataset(dataset_name, dc)
    corpus, queries, qrels = ds.corpus(), ds.queries(), ds.qrels()
    doc_ids, qids = list(corpus), list(queries)

    # Documents once, through the cache: the index is built once in the real pipeline too.
    docs = CachedEmbedder(get_embedder(model)).encode(
        [corpus[d]["text"] for d in doc_ids], "doc"
    )
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

    raw = get_embedder(model)  # uncached: every repeat is a real call
    texts = [queries[q] for q in qids]

    scores, per_query_runs, first_vecs = [], [], None
    bit_identical = True
    for _ in range(repeats):
        qv = raw.encode(texts, "query")
        if first_vecs is None:
            first_vecs = qv
        elif not np.array_equal(first_vecs, qv):
            bit_identical = False
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)

        sims = qv @ docs.T
        runs = {
            qid: [(doc_ids[j], float(sims[i][j])) for j in np.argsort(-sims[i])[:10]]
            for i, qid in enumerate(qids)
        }
        result = evaluate(runs, qrels, k=10)
        scores.append(result.mean)
        per_query_runs.append(result.per_query)

    # Per-query churn between the first two runs: how many queries move at all.
    a, b = per_query_runs[0], per_query_runs[-1]
    deltas = [abs(a[q] - b[q]) for q in a]
    return {
        "model": model_id,
        "dataset": dataset_name,
        "scores": scores,
        "range": max(scores) - min(scores),
        "bit_identical_vectors": bit_identical,
        "queries_changed": sum(d > 1e-12 for d in deltas),
        "n_queries": len(deltas),
        "max_per_query_delta": max(deltas) if deltas else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=PROBES)
    args = parser.parse_args()

    datasets = load_datasets()
    print("STRUCTURAL FLOOR — smallest expressible non-zero ΔNDCG@10")
    floors = {}
    for name in datasets.datasets:
        ds = load_dataset(name, datasets.dataset(name))
        floors[name] = structural_floor(ds.qrels(), len(ds.queries()))
        print(f"  {name:<16} n={len(ds.queries()):<4} floor = {floors[name]:.5f}")

    print(f"\nEMPIRICAL FLOOR — same queries re-embedded {args.repeats}x, documents fixed")
    print(f"{'model':<40}{'dataset':<15}{'range':>9}{'bit-ident':>11}{'q moved':>9}")
    print("-" * 84)
    records = []
    for model_id in args.models:
        for name in datasets.datasets:
            try:
                r = run(model_id, name, args.repeats)
            except Exception as exc:  # noqa: BLE001 - a dead provider must not abort the sweep
                print(f"{model_id:<40}{name:<15}  FAILED: {type(exc).__name__}: {str(exc)[:28]}")
                continue
            records.append(r)
            print(f"{r['model']:<40}{r['dataset']:<15}{r['range']:>9.5f}"
                  f"{str(r['bit_identical_vectors']):>11}"
                  f"{r['queries_changed']:>5}/{r['n_queries']:<4}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "noise_floor.json").write_text(
        json.dumps({"structural": floors, "empirical": records}, indent=2)
    )
    worst = max((r["range"] for r in records), default=0.0)
    print(f"\nworst empirical range: {worst:.5f}")
    print("structural floors    : " + ", ".join(f"{k}={v:.5f}" for k, v in floors.items()))
    print("\nReporting rule: treat any |ΔNDCG| below max(structural, empirical) as unresolvable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
