"""§5.1 correctness gate for the AILAStatutes adapter.

The count/resolve assertions are the point: if our loaded corpus differs from the one RTEB
scored, every number downstream -- including the reproduction gate -- is measuring a different
task. `validate()` failing is a hard stop, not a warning.

Marked `integration` because it downloads the pinned revision from HuggingFace.
"""

from __future__ import annotations

import pytest

from src.config import load_datasets
from src.data import load_dataset
from src.data.aila import AILAStatutes
from src.data.base import Dataset


@pytest.fixture(scope="module")
def aila() -> Dataset:
    return load_dataset("aila_statutes")


@pytest.mark.integration
def test_counts_and_qrels_resolve(aila: Dataset):
    aila.validate()


@pytest.mark.integration
def test_normalized_schema(aila: Dataset):
    qid, query = next(iter(aila.queries().items()))
    assert isinstance(qid, str) and query.strip()

    docid, doc = next(iter(aila.corpus().items()))
    assert isinstance(docid, str) and set(doc) == {"title", "text"} and doc["text"].strip()

    rels = aila.qrels()[qid]
    assert rels and all(isinstance(score, int) for score in rels.values())


@pytest.mark.integration
def test_qrels_are_the_full_labels_not_rtebs_truncated_export(aila: Dataset):
    """RTEB's relevance.jsonl holds one gold per query; the leaderboard used 2-5 (§6 note).

    The check that matters is `mean rels/query == 4.34`: because the 82-doc corpus is smaller
    than RTEB's top_k of 100, every doc is retrieved for every query, so the published
    precision@100 of 0.0434 is exactly mean_rels/100. Loading the truncated export would give
    0.0100 and put the reproduction gate permanently out of reach.
    """
    qrels = aila.qrels()
    rels_per_query = [len(rels) for rels in qrels.values()]

    assert sum(rels_per_query) == 217
    assert (min(rels_per_query), max(rels_per_query)) == (2, 5)
    assert sum(rels_per_query) / len(rels_per_query) / 100 == pytest.approx(0.0434, abs=1e-4)
    assert {score for rels in qrels.values() for score in rels.values()} == {1}


def test_borrowed_qrels_are_rejected_if_the_repos_disagree_on_the_task(tmp_path):
    """Borrowing labels across repos is only sound while the two agree on corpus and queries.

    Counts alone are not enough -- two repos can agree on 82 ids while differing in document
    text, which would silently change what the leaderboard number is measuring. So the drifted
    fixture below keeps the ids identical and changes only the text.
    """
    ours = [{"id": "d1", "text": "original statute"}]
    drifted = [{"_id": "d1", "text": "a different statute"}]

    dataset = AILAStatutes(load_datasets().dataset("aila_statutes"), cache_dir=tmp_path)
    dataset._read_jsonl = lambda filename, repo=None, revision=None: (
        drifted if repo else ours
    )

    with pytest.raises(ValueError, match="corpus differ between"):
        dataset._load_qrels()


@pytest.mark.integration
def test_second_load_hits_the_parquet_cache(aila: Dataset, tmp_path):
    fresh = type(aila)(aila.config, cache_dir=tmp_path)
    assert fresh.queries() == aila.queries()
    assert {p.name for p in (tmp_path / aila.name).iterdir()} == {"queries.parquet"}

    reloaded = type(aila)(aila.config, cache_dir=tmp_path)
    reloaded._download = _fail_on_network  # cache hit must not touch the network
    assert reloaded.queries() == aila.queries()


def _fail_on_network(*_args, **_kwargs):
    raise AssertionError("cached load must not re-download")
