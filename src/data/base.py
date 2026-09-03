"""The `Dataset` ABC and the one normalized schema all downstream code may assume (§5.1).

Per-dataset quirks -- field naming, split layout, how a "document" is assembled from a record --
are absorbed by the adapters. Nothing outside `src/data/` should know that AILAStatutes ships
statutes while ChatDoctor ships doctor answers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.config import DatasetConfig

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "datasets"


class Document(TypedDict):
    title: str
    text: str


Queries = dict[str, str]
Corpus = dict[str, Document]
Qrels = dict[str, dict[str, int]]


class Dataset(ABC):
    """Normalized retrieval dataset.

    Subclasses implement the three `_load_*` hooks; the public accessors add memoization and a
    parquet cache so repeated runs never re-hit HuggingFace.
    """

    name: str

    def __init__(self, config: DatasetConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache_dir = (cache_dir or CACHE_DIR) / self.name

    @abstractmethod
    def _load_queries(self) -> Queries: ...

    @abstractmethod
    def _load_corpus(self) -> Corpus: ...

    @abstractmethod
    def _load_qrels(self) -> Qrels: ...

    @cached_property
    def _queries(self) -> Queries:
        df = self._cached("queries", lambda: _queries_to_frame(self._load_queries()))
        return dict(zip(df["qid"], df["text"], strict=True))

    @cached_property
    def _corpus(self) -> Corpus:
        df = self._cached("corpus", lambda: _corpus_to_frame(self._load_corpus()))
        return {
            docid: Document(title=title, text=text)
            for docid, title, text in zip(df["docid"], df["title"], df["text"], strict=True)
        }

    @cached_property
    def _qrels(self) -> Qrels:
        df = self._cached("qrels", lambda: _qrels_to_frame(self._load_qrels()))
        qrels: Qrels = {}
        for qid, docid, score in zip(df["qid"], df["docid"], df["score"], strict=True):
            qrels.setdefault(qid, {})[docid] = int(score)
        return qrels

    def queries(self) -> Queries:
        return self._queries

    def corpus(self) -> Corpus:
        return self._corpus

    def qrels(self) -> Qrels:
        return self._qrels

    def validate(self) -> None:
        """Assert the §5.1 correctness gate: counts match the dataset card, qrels resolve.

        Raises rather than returning a report -- a count mismatch means we are evaluating against
        a different dataset than the leaderboard did, which invalidates every number downstream.
        """
        queries, corpus, qrels = self.queries(), self.corpus(), self.qrels()

        for label, actual, expected in (
            ("queries", len(queries), self.config.n_queries),
            ("docs", len(corpus), self.config.n_docs),
        ):
            if actual != expected:
                raise ValueError(f"{self.name}: {label} count {actual} != dataset card {expected}")

        n_qrels = sum(len(v) for v in qrels.values())
        if self.config.n_qrels is not None and n_qrels != self.config.n_qrels:
            raise ValueError(f"{self.name}: qrel count {n_qrels} != dataset card {self.config.n_qrels}")

        dangling_q = set(qrels) - set(queries)
        dangling_d = {d for rels in qrels.values() for d in rels} - set(corpus)
        if dangling_q:
            raise ValueError(f"{self.name}: {len(dangling_q)} qrel qids not in queries: {sorted(dangling_q)[:5]}")
        if dangling_d:
            raise ValueError(f"{self.name}: {len(dangling_d)} qrel docids not in corpus: {sorted(dangling_d)[:5]}")

        unjudged = set(queries) - set(qrels)
        if unjudged:
            raise ValueError(f"{self.name}: {len(unjudged)} queries have no qrels: {sorted(unjudged)[:5]}")

    def _cached(self, kind: str, build) -> pd.DataFrame:
        path = self.cache_dir / f"{kind}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df = build()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df


def _queries_to_frame(queries: Queries) -> pd.DataFrame:
    return pd.DataFrame(
        {"qid": list(queries), "text": list(queries.values())}, dtype="object"
    )


def _corpus_to_frame(corpus: Corpus) -> pd.DataFrame:
    return pd.DataFrame(
        [{"docid": k, "title": v["title"], "text": v["text"]} for k, v in corpus.items()],
        columns=["docid", "title", "text"],
        dtype="object",
    )


def _qrels_to_frame(qrels: Qrels) -> pd.DataFrame:
    rows = [
        {"qid": qid, "docid": docid, "score": score}
        for qid, rels in qrels.items()
        for docid, score in rels.items()
    ]
    return pd.DataFrame(rows, columns=["qid", "docid", "score"])
