"""Retrieval-mode tests (§5.4).

`rrf_fuse` is the offline cross-check for turbopuffer's server-side fusion, so it gets real unit
tests: it is only useful as a check if it is independently correct.
"""

from __future__ import annotations

import os

import pytest

from src.config import load_datasets, load_experiment
from src.eval.metrics import evaluate
from src.index.tpuf import Index
from src.retrieve.bm25 import bm25_batch
from src.retrieve.hybrid import rrf_fuse


def _ranking(*doc_ids: str):
    return [(d, float(len(doc_ids) - i)) for i, d in enumerate(doc_ids)]


def test_rrf_rewards_agreement_across_arms():
    """A doc both arms rank 2nd should beat one that is 1st in a single arm."""
    dense = _ranking("only_dense", "agreed", "c")
    lexical = _ranking("only_lexical", "agreed", "c")
    fused = dict(rrf_fuse([dense, lexical], k_rrf=60))
    assert fused["agreed"] > fused["only_dense"]
    assert fused["agreed"] > fused["only_lexical"]


def test_rrf_uses_rank_not_score():
    """RRF is rank-based, so an arm's score scale must not matter."""
    a = [("x", 1000.0), ("y", 999.0)]
    b = [("x", 0.001), ("y", 0.0001)]
    assert rrf_fuse([a], k_rrf=60) == rrf_fuse([b], k_rrf=60)


def test_rrf_is_deterministic_under_ties():
    """Equal fused scores break by doc id, so a run is reproducible (§9)."""
    left = rrf_fuse([_ranking("b", "a")], k_rrf=60)
    right = rrf_fuse([_ranking("b", "a")], k_rrf=60)
    assert left == right


def test_rrf_k_damps_the_weight_of_top_ranks():
    """Small k sharpens the advantage of rank 1; large k flattens it."""
    ranking = [_ranking("first", "second")]
    sharp = dict(rrf_fuse(ranking, k_rrf=1))
    flat = dict(rrf_fuse(ranking, k_rrf=1000))
    assert sharp["first"] / sharp["second"] > flat["first"] / flat["second"]


def test_rrf_truncates_to_k():
    fused = rrf_fuse([_ranking(*[f"d{i}" for i in range(50)])], k=10)
    assert len(fused) == 10


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TURBOPUFFER_API_KEY"), reason="needs TURBOPUFFER_API_KEY")
def test_bm25_is_a_floor_not_a_failure():
    """BM25 must actually retrieve — a near-zero score would mean the FTS schema is broken.

    It is the study's lineage-neutral control (§5.5), so "it returned nothing" and "lexical
    matching is genuinely hard here" must not be confusable.
    """
    dataset_config = load_datasets().dataset("aila_statutes")
    model = load_experiment().model("gemini-embedding-001")
    index = Index("aila_statutes", dataset_config, model)
    if not index.exists():
        pytest.skip("index not built")

    from src.data import load_dataset

    dataset = load_dataset("aila_statutes", dataset_config)
    queries, qrels = dataset.queries(), dataset.qrels()
    runs = bm25_batch(index, list(queries), queries, k=10)

    assert all(runs.values()), "every query must retrieve something"
    score = evaluate(runs, qrels, k=10).mean
    assert 0.05 < score < 0.45, f"BM25 {score:.4f} outside the plausible floor band"
