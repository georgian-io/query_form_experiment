"""Retrieve + evaluate, and check the result against RTEB's published number (§8 Phase 0, §9).

This is the reproduction gate. It runs the ORIGINAL human queries -- no transform -- because its
whole purpose is to establish that the harness measures what RTEB measured. Until it passes, a
ΔNDCG from a transformed query set cannot be distinguished from a harness bug.

    uv run python scripts/03_run_retrieval.py --dataset aila_statutes --model gemini-embedding-001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_hash, load_datasets, load_experiment  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.index.embed import CachedEmbedder, get_embedder  # noqa: E402
from src.index.tpuf import Index  # noqa: E402
from src.retrieve.bm25 import bm25_batch  # noqa: E402
from src.retrieve.dense import dense_batch  # noqa: E402
from src.retrieve.hybrid import hybrid_batch  # noqa: E402
from src.transform.cache import flatten_id  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# §9 gate tolerance. At n=50 a single query changing rank moves NDCG@10 by ~0.02, so a band
# tighter than this would chase noise rather than catch harness bugs.
PASS_TOLERANCE = 0.01
INVESTIGATE_TOLERANCE = 0.02

_DIAGNOSIS = """
Diagnose in this order (§10, ordered by how much each can move the number):
  1. Query/doc instruction -- for gemini-embedding-001, task_type must be RETRIEVAL_QUERY on
     queries and RETRIEVAL_DOCUMENT on docs. RTEB passes no text prefix at all.
  2. Pooling -- N/A for hosted APIs; check it first for any Novita model.
  3. Normalization & similarity -- 3072d gemini output is already unit-norm; cosine.
  4. Tokenizer -- only affects BM25, so irrelevant to a dense-only mismatch.
Also check: are the qrels the full 217 pairs, or RTEB's truncated 50-row export?
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=None, help="defaults to experiment.yaml k")
    parser.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--condition", default="human",
                        help="'human' for original queries, else a transform condition")
    parser.add_argument("--generator", default=None,
                        help="required when --condition is not 'human'")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    experiment = load_experiment()
    dataset_config = load_datasets().dataset(args.dataset)
    model = experiment.model(args.model)
    k = args.k or experiment.k

    dataset = load_dataset(args.dataset, dataset_config)
    dataset.validate()
    queries, qrels = dataset.queries(), dataset.qrels()

    if args.condition != "human":
        queries = _load_transformed(args.dataset, args.condition, args.generator, queries)

    qids = list(queries)
    index = Index(args.dataset, dataset_config, model)
    if not index.exists():
        print(f"namespace {index.name} does not exist; run scripts/01_build_index.py first")
        return 1

    # BM25 is purely lexical, so it needs no query vectors at all -- skipping the encode keeps
    # the lexical control genuinely independent of the embedding provider.
    query_vectors = None
    if args.mode in ("dense", "hybrid"):
        embedder = CachedEmbedder(get_embedder(model))
        print(f"encoding {len(qids)} queries with {model.id} ({model.query_task_type})...")
        query_vectors = embedder.encode(
            [queries[qid] for qid in qids], "query", batch_size=args.batch_size
        )

    print(f"retrieving top-{k} from {index.name} ({args.mode})...")
    if args.mode == "dense":
        runs = dense_batch(index, qids, query_vectors, k=k)
    elif args.mode == "bm25":
        runs = bm25_batch(index, qids, queries, k=k)
    else:
        runs = hybrid_batch(
            index, qids, query_vectors, queries, k=k, depth=experiment.retrieve_depth
        )
    result = evaluate(runs, qrels, k=k)

    # The leaderboard is a dense-retrieval board, so only dense has a number to reproduce.
    # Only original queries have a published number; a transformed run is measured against our
    # own human baseline instead, which is the whole point of §9's gate coming first.
    published = (
        dataset_config.published_ndcg_at_10.get(model.id)
        if args.mode == "dense" and args.condition == "human"
        else None
    )
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "model": model.id,
        "retrieval_mode": args.mode,
        "condition": args.condition,
        "generator": args.generator,
        "metric": result.metric,
        "score": result.mean,
        "n_queries": result.n_queries,
        "published": published,
        "config_hash": config_hash(model, dataset_config),
        "per_query": result.per_query,
    }
    _write(record, args.dataset, model.id, args.mode, args.condition, args.generator)

    print(f"\n{result.metric} = {result.mean:.5f}  (n={result.n_queries})")
    return _report_gate(result.mean, published)


def _report_gate(score: float, published: float | None) -> int:
    if published is None:
        print("no published RTEB number for this mode/model -- gate not evaluated")
        return 0

    delta = score - published
    print(f"published    = {published:.5f}")
    print(f"delta        = {delta:+.5f}  ({delta / published:+.2%})")

    if abs(delta) <= PASS_TOLERANCE:
        print(f"\nGATE PASSED (|delta| <= {PASS_TOLERANCE})")
        return 0
    verdict = "INVESTIGATE" if abs(delta) <= INVESTIGATE_TOLERANCE else "FAILED"
    print(f"\nGATE {verdict} (|delta| > {PASS_TOLERANCE})")
    print(_DIAGNOSIS)
    return 1


def _load_transformed(dataset: str, condition: str, generator: str | None, original: dict) -> dict:
    """Swap in a transformed query set, keeping the qids aligned with the fixed qrels."""
    if generator is None:
        raise SystemExit("--generator is required when --condition is not 'human'")
    path = RESULTS_DIR / f"queries__{dataset}__{condition}__{flatten_id(generator)}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/02_run_transform.py first")

    payload = json.loads(path.read_text())
    queries = payload["queries"]
    # A transformed set may deliberately be a subsample (--sample), so it need not cover every
    # query -- but every qid it DOES carry must be a real query, or the qrels would misalign.
    unknown = set(queries) - set(original)
    if unknown:
        raise SystemExit(
            f"{path.name} contains {len(unknown)} qids not in {dataset} "
            f"(e.g. {sorted(unknown)[:3]}); the query set does not match the dataset"
        )
    return dict(queries)


def _write(record: dict, dataset: str, model_id: str, mode: str,
           condition: str, generator: str | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Open-weight ids carry an org prefix (`BAAI/bge-base-en-v1.5`); flatten it so the id stays
    # one path segment. Same substitution as the turbopuffer namespace, for the same reason.
    suffix = "" if condition == "human" else f"__{condition}__{flatten_id(generator)}"
    path = RESULTS_DIR / f"{mode}__{dataset}__{model_id.replace('/', '__')}{suffix}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
