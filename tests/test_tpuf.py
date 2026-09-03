"""turbopuffer schema and round-trip tests.

The schema test is a unit test because §9 makes the tokenizer a pinned scoring parameter: it must
be asserted in CI, not only when someone happens to run against a live cluster.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from src.config import load_datasets, load_experiment
from src.data.base import Document
from src.index.tpuf import Index, build_schema, namespace_name


@pytest.fixture
def model():
    return load_experiment().model("gemini-embedding-001")


@pytest.fixture
def dataset_config():
    return load_datasets().dataset("aila_statutes")


def test_namespace_convention(model):
    assert namespace_name("aila_statutes", model.id) == "pilot__aila_statutes__gemini-embedding-001"


def test_schema_pins_the_tokenizer_explicitly(model, dataset_config):
    """§9: every FTS setting is written out, so an upstream default cannot move BM25 silently."""
    fts = build_schema(model, dataset_config.tokenizer)["text"]["full_text_search"]
    assert fts == {
        "tokenizer": "word_v4",
        "language": "english",
        "stemming": True,
        "remove_stopwords": True,
        "case_sensitive": False,
        "ascii_folding": True,
        # Scoring constants, pinned rather than inherited from the server (§9).
        "k1": 1.2,
        "b": 0.75,
        "max_token_length": 39,
    }


def test_schema_vector_dim_follows_the_model(model, dataset_config):
    vector = build_schema(model, dataset_config.tokenizer)["vector"]
    assert vector["type"] == f"[{model.dim}]f32"
    assert vector["ann"] is True


def test_code_datasets_would_preserve_identifiers(model, dataset_config):
    """§5.3: stemming an identifier destroys it, so code corpora must opt out."""
    code_tokenizer = dataset_config.tokenizer.model_copy(
        update={"stemming": False, "remove_stopwords": False, "case_sensitive": True}
    )
    fts = build_schema(model, code_tokenizer)["text"]["full_text_search"]
    assert (fts["stemming"], fts["case_sensitive"]) == (False, True)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TURBOPUFFER_API_KEY"), reason="needs TURBOPUFFER_API_KEY")
def test_upsert_and_dense_round_trip(model, dataset_config):
    """Writes a 3-doc namespace and checks the nearest neighbour comes back first."""
    tiny = model.model_copy(update={"dim": 4})
    index = Index(f"smoke_{uuid.uuid4().hex[:8]}", dataset_config, tiny)

    corpus = {
        "a": Document(title="", text="criminal breach of trust by a public servant"),
        "b": Document(title="", text="dowry prohibition and related offences"),
        "c": Document(title="", text="procedure for arrest without warrant"),
    }
    doc_ids = list(corpus)
    vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32
    )

    try:
        assert index.upsert(corpus, vectors, doc_ids) == 3

        ranking = index.dense(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), top_k=3)
        assert [doc_id for doc_id, _ in ranking] == ["a", "b", "c"]

        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True), "scores must be higher-is-better"
    finally:
        index.delete()
