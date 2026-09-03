"""BM25 retrieval (§5.4).

BM25 is not a baseline for completeness — it is the study's lineage-neutral control. Every
embedding model on the slate was trained on data an LLM had a hand in, so if LLM-generated
queries help all of them, a purely lexical scorer is the only thing that can tell "the queries
got genuinely better" apart from "the queries moved toward the models' training distribution"
(§5.5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.eval.metrics import Ranking, Runs
from src.index.tpuf import Index

# turbopuffer rejects a BM25 query longer than 8192 Unicode code points with a 400. A rare
# over-long rewrite (e.g. a "terse" generation that failed to compress -- one ChatDoctor query came
# out at 11,770 chars against a median of 88) would otherwise kill the whole lexical run. Truncate
# defensively: this touches only the BM25 CONTROL and only the handful of queries over the limit,
# scoring them on their first 8192 code points rather than dropping them.
_BM25_MAX_CODEPOINTS = 8192


def bm25(index: Index, query_text: str, k: int = 10) -> Ranking:
    return index.bm25(query_text[:_BM25_MAX_CODEPOINTS], top_k=k)


def bm25_batch(index: Index, qids: Sequence[str], queries: Mapping[str, str], k: int = 10) -> Runs:
    return {qid: bm25(index, queries[qid], k=k) for qid in qids}
