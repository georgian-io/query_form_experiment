"""Dense retrieval (§5.4).

Deliberately blind to where a query string came from: the same function serves human queries in
Phase 0 and transformed queries in Phase 2+. That is the layering rule from §2 -- if this module
ever needs to know about a transform condition, the boundary has been broken.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.eval.metrics import Ranking, Runs
from src.index.tpuf import Index


def dense(index: Index, query_vector: np.ndarray, k: int = 10) -> Ranking:
    return index.dense(query_vector, top_k=k)


def dense_batch(
    index: Index,
    qids: Sequence[str],
    query_vectors: np.ndarray,
    k: int = 10,
) -> Runs:
    if len(qids) != len(query_vectors):
        raise ValueError(f"{len(qids)} qids but {len(query_vectors)} vectors")
    return {qid: dense(index, vector, k=k) for qid, vector in zip(qids, query_vectors, strict=True)}
