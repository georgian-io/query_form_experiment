"""NDCG@10 tests against hand-computable cases.

AILAStatutes has exactly one binary-relevant doc per query, so NDCG@10 collapses to
1/log2(rank+1) -- worth asserting directly, because it is the arithmetic the reproduction gate
depends on.
"""

from __future__ import annotations

import math

import pytest

from src.eval.metrics import evaluate


def _ranking(*doc_ids: str):
    """Descending scores, so list order is rank order."""
    return [(d, float(len(doc_ids) - i)) for i, d in enumerate(doc_ids)]


def test_single_relevant_doc_scores_by_reciprocal_log_rank():
    qrels = {"q1": {"gold": 1}}
    for rank, expected in [(1, 1.0), (2, 1 / math.log2(3)), (10, 1 / math.log2(11))]:
        docs = [f"d{i}" for i in range(rank - 1)] + ["gold"]
        result = evaluate({"q1": _ranking(*docs)}, qrels, k=10)
        assert result.mean == pytest.approx(expected), f"rank {rank}"


def test_relevant_doc_below_the_cutoff_scores_zero():
    qrels = {"q1": {"gold": 1}}
    docs = [f"d{i}" for i in range(10)] + ["gold"]
    assert evaluate({"q1": _ranking(*docs)}, qrels, k=10).mean == 0.0


def test_attempted_but_empty_query_scores_zero():
    """A retrieval failure -- attempted, returned nothing -- scores 0 via its empty ranking.

    In a full run every query is passed to retrieval, so a failure is a present-but-empty entry,
    not an absent one. That path still scores 0 and cannot raise the mean.
    """
    qrels = {"q1": {"gold": 1}, "q2": {"gold2": 1}}
    result = evaluate({"q1": _ranking("gold"), "q2": []}, qrels, k=10)
    assert result.per_query == {"q1": 1.0, "q2": 0.0}
    assert result.mean == 0.5


def test_unattempted_query_is_excluded_not_scored_zero():
    """A subsample scores only what it retrieved; un-sampled queries are absent from `runs`.

    Iterating `qrels` instead of `runs` scored the un-sampled queries 0, collapsing a
    1,000-of-5,591 subsample mean toward zero (0.72 -> 0.13 in practice).
    """
    qrels = {f"q{i}": {"gold": 1} for i in range(5)}
    result = evaluate({"q0": _ranking("gold"), "q1": _ranking("x", "gold")}, qrels, k=10)
    assert set(result.per_query) == {"q0", "q1"}
    assert result.n_queries == 2  # not 5
    assert result.mean == pytest.approx((1.0 + 1 / math.log2(3)) / 2)


def test_empty_ranking_scores_zero():
    result = evaluate({"q1": []}, {"q1": {"gold": 1}}, k=10)
    assert result.per_query == {"q1": 0.0}


def test_graded_relevance_rewards_ordering():
    """Guards the ΔNDCG path for datasets whose qrels are not binary."""
    qrels = {"q1": {"a": 2, "b": 1}}
    better = evaluate({"q1": _ranking("a", "b")}, qrels, k=10).mean
    worse = evaluate({"q1": _ranking("b", "a")}, qrels, k=10).mean
    assert better == 1.0
    assert worse < better


def test_per_query_scores_are_emitted_not_just_the_mean():
    """§5.5: ΔNDCG attribution needs per-query scores."""
    qrels = {"q1": {"gold": 1}, "q2": {"gold": 1}}
    runs = {"q1": _ranking("gold"), "q2": _ranking("x", "gold")}
    result = evaluate(runs, qrels, k=10)
    assert result.per_query["q1"] == 1.0
    assert result.per_query["q2"] == pytest.approx(1 / math.log2(3))
    assert result.metric == "NDCG@10"


def test_scores_beyond_k_do_not_affect_the_cutoff_metric():
    qrels = {"q1": {"gold": 1}}
    short = evaluate({"q1": _ranking("gold")}, qrels, k=10)
    padded = evaluate({"q1": _ranking("gold", *[f"d{i}" for i in range(50)])}, qrels, k=10)
    assert short.mean == padded.mean
