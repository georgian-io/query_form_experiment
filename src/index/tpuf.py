"""turbopuffer namespace management, schema, upsert, and dense retrieval (§5.3, §5.4).

One namespace per (dataset x embedding model), holding the dense vector and the BM25 text on the
*same* rows. That layout follows from the experiment's central asymmetry: corpus and qrels are
fixed across every query condition, so a corpus is indexed once and queried many times. No code
path should ever re-index because the query condition changed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

import numpy as np
from turbopuffer import Turbopuffer
from turbopuffer.types import NamespaceQueryResponse

from src.config import DatasetConfig, EmbeddingModelConfig, Similarity, TokenizerConfig
from src.data.base import Corpus

_DISTANCE_METRIC = {
    Similarity.COSINE: "cosine_distance",
    Similarity.DOT: "euclidean_squared",
}


def namespace_name(dataset: str, model_id: str) -> str:
    """§5.3 convention: `pilot__{dataset}__{embed_model}`.

    Model ids are the provider's own strings, so open-weight ones carry an org prefix
    (`BAAI/bge-base-en-v1.5`). The slash is replaced rather than stripped, so two models that
    differ only in org cannot collide onto one namespace.
    """
    return f"pilot__{dataset}__{model_id.replace('/', '__')}"


def _client() -> Turbopuffer:
    api_key = os.environ.get("TURBOPUFFER_API_KEY")
    if not api_key:
        raise RuntimeError("TURBOPUFFER_API_KEY is not set; see .env.example")
    region = os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1")
    return Turbopuffer(api_key=api_key, region=region)


def build_schema(model: EmbeddingModelConfig, tokenizer: TokenizerConfig) -> dict:
    """Explicit schema for both retrieval modes.

    Every full-text-search setting is written out rather than defaulted, per §9 -- the tokenizer
    is a scoring parameter, and an upstream default change would move BM25 numbers between runs
    with nothing in the diff to show for it.
    """
    return {
        "vector": {
            "type": f"[{model.dim}]f32",
            "ann": True,
        },
        "text": {
            "type": "string",
            "full_text_search": tokenizer.model_dump(exclude_none=True),
        },
    }


class Index:
    """A single (dataset x model) turbopuffer namespace."""

    def __init__(
        self,
        dataset_name: str,
        dataset_config: DatasetConfig,
        model: EmbeddingModelConfig,
        client: Turbopuffer | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.model = model
        self.name = namespace_name(dataset_name, model.id)
        self._client = client or _client()
        self._ns = self._client.namespace(self.name)

    @property
    def distance_metric(self) -> str:
        return _DISTANCE_METRIC[self.model.similarity]

    def exists(self) -> bool:
        return self._ns.exists()

    def delete(self) -> None:
        if self.exists():
            self._ns.delete_all()

    def upsert(self, corpus: Corpus, vectors: np.ndarray, doc_ids: Sequence[str],
               batch_size: int = 1000) -> int:
        """Write documents and their vectors.

        `doc_ids` is passed alongside `corpus` rather than derived from it because it fixes the
        row order that `vectors` was encoded in -- pairing a vector with the wrong document is a
        failure no downstream metric could distinguish from a bad model.

        Each batch is one synchronous `write` round-trip, so the batch size sets the number of
        HTTP calls: at 200 the 244,600-doc CUREv1 corpus took ~1,220 writes / ~44 min. 1000 cuts
        that 5x and stays well within turbopuffer's per-request limit (~256MB) even for a 4096-d
        vector. It was invisibly fine before only because every prior corpus was <= 5,545 docs.
        """
        if len(doc_ids) != len(vectors):
            raise ValueError(f"{len(doc_ids)} ids but {len(vectors)} vectors")

        schema = build_schema(self.model, self.dataset_config.tokenizer)
        written = 0
        for batch in _batched(list(zip(doc_ids, vectors, strict=True)), batch_size):
            rows = [
                {
                    "id": doc_id,
                    "vector": vector.tolist(),
                    "text": _index_text(corpus[doc_id]),
                }
                for doc_id, vector in batch
            ]
            self._ns.write(
                upsert_rows=rows,
                distance_metric=self.distance_metric,
                schema=schema,
            )
            written += len(rows)
        return written

    def dense(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """ANN retrieval. Returns `(doc_id, score)` ordered best-first (§5.4)."""
        response = self._ns.query(
            rank_by=("vector", "ANN", query_vector.tolist()),
            top_k=top_k,
            include_attributes=False,
        )
        return _to_ranking(response, higher_is_better=False)

    def bm25(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Lexical retrieval over the same rows the dense vectors live on (§5.4).

        Takes the *raw* query text. An embedding model's query instruction is a signal to that
        model, not vocabulary — feeding "Represent the query for retrieving supporting
        documents:" to BM25 would just add constant tokens to every query and dilute the real
        ones.
        """
        response = self._ns.query(
            rank_by=("text", "BM25", query_text),
            top_k=top_k,
            include_attributes=False,
        )
        return _to_ranking(response, higher_is_better=True)

    def hybrid(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        depth: int = 100,
    ) -> list[tuple[str, float]]:
        """Server-side RRF fusion of the dense and BM25 sub-queries (§5.4).

        Sub-queries retrieve to `depth` rather than `top_k` so fusion sees a stable tail: a
        document ranked 40th by BM25 and 3rd by vector should be able to surface, which it cannot
        if each arm is truncated at 10 before fusing.
        """
        response = self._ns.multi_query(
            queries=[
                {"rank_by": ("vector", "ANN", query_vector.tolist()), "limit": depth},
                {"rank_by": ("text", "BM25", query_text), "limit": depth},
            ],
            rerank_by=("RRF",),
        )
        fused = response.results[0]
        return [
            (str(row.id), float(row.model_extra.get("$dist", 0.0)))
            for row in (fused.rows or [])
        ][:top_k]

    def branch(self, branch_name: str) -> None:
        """Copy-on-write snapshot of a finished index -- a citable, re-runnable state (§9)."""
        self._client.namespace(branch_name).write(
            branch_from_namespace={"namespace": self.name}
        )


def _index_text(doc: dict) -> str:
    """The string BM25 sees.

    Title is prepended only when present; AILAStatutes ships empty titles with the statute name
    folded into `text`, and injecting a stray separator would change the token stream.
    """
    title = doc.get("title", "").strip()
    return f"{title}\n{doc['text']}" if title else doc["text"]


def _to_ranking(
    response: NamespaceQueryResponse, *, higher_is_better: bool
) -> list[tuple[str, float]]:
    """Convert turbopuffer rows into descending relevance scores.

    turbopuffer reports both under the same `$dist` key, but the sense differs by rank mode: it
    is a *distance* for vector ranking (lower is better) and a *score* for BM25 (higher is
    better). pytrec_eval wants higher-is-better throughout, so callers declare the sense and the
    sign is normalized here rather than at each call site. The value arrives as a pydantic extra
    because `$dist` is not a valid Python identifier.
    """
    sign = 1.0 if higher_is_better else -1.0
    return [
        (str(row.id), sign * float(row.model_extra["$dist"])) for row in response.rows or []
    ]


def _batched(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
