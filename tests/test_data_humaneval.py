"""§5.1 correctness gate for the HumanEval adapter, plus its code-tokenizer settings.

The tokenizer assertions are unit tests, not integration ones: they encode *why* code needs
different settings, and that reasoning should be checked in CI rather than only when someone
runs against a live cluster.
"""

from __future__ import annotations

import pytest

from src.config import load_datasets, load_experiment
from src.data import load_dataset
from src.data.base import Dataset
from src.index.tpuf import build_schema


@pytest.fixture(scope="module")
def humaneval() -> Dataset:
    return load_dataset("humaneval")


def test_code_tokenizer_preserves_identifiers():
    """§5.3: stemming turns `enumerate` into `enumer`; `in`/`for` are keywords, not noise."""
    fts = build_schema(
        load_experiment().model("gemini-embedding-001"),
        load_datasets().dataset("humaneval").tokenizer,
    )["text"]["full_text_search"]

    assert fts["stemming"] is False
    assert fts["remove_stopwords"] is False
    assert fts["case_sensitive"] is True
    assert fts["tokenizer"] == "word_v4"


def test_bm25_constants_are_pinned_not_inherited():
    """§9: turbopuffer fills k1/b/max_token_length server-side, and they scale every BM25 score.

    Reading the stored schema back off a live namespace showed these arriving as defaults we had
    never written down -- which is the "rely on the default" the plan forbids.
    """
    for name in ("aila_statutes", "humaneval"):
        fts = build_schema(
            load_experiment().model("gemini-embedding-001"),
            load_datasets().dataset(name).tokenizer,
        )["text"]["full_text_search"]
        assert (fts["k1"], fts["b"], fts["max_token_length"]) == (1.2, 0.75, 39), name
        # turbopuffer defaults this even for code, so it is pinned rather than left to chance.
        assert fts["language"] == "english", name


def test_code_and_prose_tokenizers_actually_differ():
    """A single shared tokenizer would silently be wrong for one of the two dataset kinds."""
    datasets = load_datasets()
    prose = datasets.dataset("aila_statutes").tokenizer
    code = datasets.dataset("humaneval").tokenizer
    assert (prose.stemming, prose.remove_stopwords) == (True, True)
    assert (code.stemming, code.remove_stopwords) == (False, False)


def test_humaneval_forbids_hyde():
    """The docstring is the spec, so a hypothetical answer changes the task, not the query (§6)."""
    config = load_datasets().dataset("humaneval")
    assert "hyde" not in [c.value for c in config.priority_transforms]
    assert config.qrel_risk == "high"


@pytest.mark.integration
def test_counts_and_qrels_resolve(humaneval: Dataset):
    humaneval.validate()


@pytest.mark.integration
def test_relevance_is_genuinely_one_to_one(humaneval: Dataset):
    """Unlike AILAStatutes, this packaging is not truncated -- 158 problems, one solution each.

    RTEB's published precision@100 = 0.01 and recall@100 = 1.0 are exactly what a single
    relevant document per query implies, so the 1:1 shape is confirmed rather than assumed.
    """
    qrels = humaneval.qrels()
    assert {len(rels) for rels in qrels.values()} == {1}
    assert len(qrels) == 158


@pytest.mark.integration
def test_queries_are_specs_and_docs_are_code(humaneval: Dataset):
    corpus = humaneval.corpus()
    assert all(doc["title"] == "" for doc in corpus.values()), "this packaging has no titles"
    # Solutions are Python bodies; indentation is the cheapest reliable signal that we did not
    # accidentally load the docstrings on both sides.
    assert sum(doc["text"].startswith((" ", "\t")) for doc in corpus.values()) > 100
