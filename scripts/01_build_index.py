"""Encode a dataset's corpus and upsert it into its turbopuffer namespace (§5.3).

Indexing is per (dataset x embedding model) and happens exactly once: corpus and qrels are fixed
across every query condition, so nothing downstream ever needs a rebuild. Embeddings are cached,
so a rerun after an interrupted upsert costs nothing.

    uv run python scripts/01_build_index.py --dataset aila_statutes --model gemini-embedding-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_datasets, load_experiment  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.index.embed import CachedEmbedder, get_embedder  # noqa: E402
from src.index.tpuf import Index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--force", action="store_true", help="re-upsert even if the namespace exists"
    )
    args = parser.parse_args()

    dataset_config = load_datasets().dataset(args.dataset)
    model = load_experiment().model(args.model)

    dataset = load_dataset(args.dataset, dataset_config)
    dataset.validate()
    corpus = dataset.corpus()
    print(f"{args.dataset}: {len(dataset.queries())} queries, {len(corpus)} docs (counts verified)")

    index = Index(args.dataset, dataset_config, model)
    if index.exists() and not args.force:
        print(f"namespace {index.name} already exists; pass --force to re-upsert")
        return 0

    doc_ids = list(corpus)
    embedder = CachedEmbedder(get_embedder(model))
    # RTEB embeds the document text alone; `title` is empty here and it never reads it anyway.
    texts = [corpus[doc_id]["text"] for doc_id in doc_ids]

    print(f"encoding {len(texts)} docs with {model.id} ({model.dim}d, {model.doc_task_type})...")
    vectors = embedder.encode(texts, "doc", batch_size=args.batch_size)

    written = index.upsert(corpus, vectors, doc_ids)
    print(f"upserted {written} rows into {index.name} (metric: {index.distance_metric})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
