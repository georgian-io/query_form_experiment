"""§5.1 correctness gate for the ChatDoctor adapter.

The `_id` test is the one worth having: this is the only dataset of the three that follows
mteb's field name rather than RTEB's, and getting it wrong would not raise -- it would produce
an empty or misaligned corpus.
"""

from __future__ import annotations

import pytest

from src.config import load_datasets
from src.data import load_dataset
from src.data.base import Dataset
from src.data.rteb import RTEBJsonlDataset


@pytest.fixture(scope="module")
def chatdoctor() -> Dataset:
    return load_dataset("chatdoctor")


def test_id_field_accepts_both_conventions():
    """AILAStatutes/HumanEval use `id`; ChatDoctor uses `_id`. Both must load."""
    assert RTEBJsonlDataset._row_id({"id": "a", "text": "x"}) == "a"
    assert RTEBJsonlDataset._row_id({"_id": "b", "text": "x"}) == "b"


def test_missing_id_field_raises_rather_than_inventing_one():
    """A silently invented id would misalign every qrel."""
    with pytest.raises(KeyError, match="neither 'id' nor '_id'"):
        RTEBJsonlDataset._row_id({"docid": "c", "text": "x"})


def test_corpus_size_is_the_packaged_subset_not_the_card_figure():
    """§6 quotes 112k docs; RTEB packaged 5,545. The smaller figure is what we score against."""
    cfg = load_datasets().dataset("chatdoctor")
    assert (cfg.n_queries, cfg.n_docs, cfg.n_qrels) == (5591, 5545, 5591)
    assert cfg.qrel_risk == "high"


def test_prose_tokenizer_not_the_code_one():
    """Patient narrative is prose: stemming and stopword removal both apply."""
    tok = load_datasets().dataset("chatdoctor").tokenizer
    assert (tok.stemming, tok.remove_stopwords, tok.case_sensitive) == (True, True, False)


@pytest.mark.integration
def test_counts_and_qrels_resolve(chatdoctor: Dataset):
    chatdoctor.validate()


@pytest.mark.integration
def test_single_gold_per_query(chatdoctor: Dataset):
    """Published precision@100 = recall@100/100 implies exactly one relevant doc per query."""
    qrels = chatdoctor.qrels()
    assert {len(v) for v in qrels.values()} == {1}
    assert len(qrels) == 5591


@pytest.mark.integration
def test_corpus_is_answers_and_queries_are_narratives(chatdoctor: Dataset):
    """Guards against loading the same side twice -- answers are not first-person questions."""
    queries, corpus = chatdoctor.queries(), chatdoctor.corpus()
    assert all(d["title"] == "" for d in corpus.values()), "this packaging has no titles"
    # Patient narratives are first-person; doctors' answers overwhelmingly are not.
    first_person = sum(q.lower().startswith(("hi", "hello", "i ", "my ")) for q in queries.values())
    assert first_person > len(queries) * 0.2
