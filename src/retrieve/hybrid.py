"""Hybrid retrieval via turbopuffer's server-side RRF (§5.4).

Fusion config is frozen across every run (§9, §10): RRF is rank-based, so changing its parameters
between conditions would move scores for reasons unrelated to the queries — which is precisely
the confound this study measures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.eval.metrics import Ranking, Runs
from src.index.tpuf import Index


def hybrid(
    index: Index,
    query_vector: np.ndarray,
    query_text: str,
    k: int = 10,
    depth: int = 100,
) -> Ranking:
    return index.hybrid(query_vector, query_text, top_k=k, depth=depth)


def hybrid_batch(
    index: Index,
    qids: Sequence[str],
    query_vectors: np.ndarray,
    queries: Mapping[str, str],
    k: int = 10,
    depth: int = 100,
) -> Runs:
    if len(qids) != len(query_vectors):
        raise ValueError(f"{len(qids)} qids but {len(query_vectors)} vectors")
    return {
        qid: hybrid(index, vector, queries[qid], k=k, depth=depth)
        for qid, vector in zip(qids, query_vectors, strict=True)
    }


def rrf_fuse(rankings: Sequence[Ranking], k_rrf: int = 60, k: int = 10) -> Ranking:
    """Offline RRF, for cross-checking turbopuffer's server-side fusion (§5.4).

    Independent implementation on purpose: it is the parity check, so it must not share code
    with the thing it checks.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:k]
