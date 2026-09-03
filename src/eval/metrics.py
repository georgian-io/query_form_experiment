"""NDCG@10 via pytrec_eval against fixed qrels (§5.5).

Per-query scores are the primary output and the mean is derived from them, not the other way
round. ΔNDCG attribution (§5.5) regresses per-query deltas on query features, so a mean-only
metric would foreclose the analysis the whole study is built to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import pytrec_eval

from src.data.base import Qrels

Ranking = list[tuple[str, float]]
Runs = dict[str, Ranking]


@dataclass(frozen=True)
class EvalResult:
    metric: str
    per_query: dict[str, float]

    @property
    def mean(self) -> float:
        return mean(self.per_query.values()) if self.per_query else 0.0

    @property
    def n_queries(self) -> int:
        return len(self.per_query)


def evaluate(runs: Runs, qrels: Qrels, k: int = 10, metric: str = "ndcg") -> EvalResult:
    """Score `runs` against `qrels`, over the queries that were actually attempted.

    Scored set = the queries present in `runs` that also have labels. A retrieval failure is an
    *attempted* query that returned nothing -- it is still a key in `runs` (with an empty
    ranking), so it is scored 0 and cannot raise the mean by shrinking the denominator. But a
    query that was never attempted -- e.g. the 4,591 queries outside a 1,000-query subsample --
    must NOT be scored 0, or the mean collapses. Iterating `qrels` instead of `runs` did exactly
    that. For a full run `runs` covers every qrel, so the two are identical; only the subsample
    case differs.
    """
    measure = f"{metric}_cut.{k}"
    evaluator = pytrec_eval.RelevanceEvaluator(_to_pytrec_qrels(qrels), {measure})

    scored = evaluator.evaluate(_to_pytrec_run(runs, k))
    attempted = [qid for qid in runs if qid in qrels]
    per_query = {
        qid: scored.get(qid, {}).get(f"{metric}_cut_{k}", 0.0) for qid in attempted
    }
    return EvalResult(metric=f"{metric.upper()}@{k}", per_query=per_query)


def _to_pytrec_qrels(qrels: Qrels) -> dict[str, dict[str, int]]:
    return {
        qid: {docid: int(score) for docid, score in rels.items()} for qid, rels in qrels.items()
    }


def _to_pytrec_run(runs: Runs, k: int) -> dict[str, dict[str, float]]:
    """Truncate to k and drop empty rankings.

    pytrec_eval rejects a query whose run is empty, so those are omitted here and restored as
    explicit zeros by `evaluate`.
    """
    truncated = {qid: dict(ranking[:k]) for qid, ranking in runs.items()}
    return {qid: ranking for qid, ranking in truncated.items() if ranking}
