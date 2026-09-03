"""§5.1 correctness gate for the TREC-COVID adapter.

TREC-COVID is not RTEB-packaged, so this adapter carries the most bespoke logic on the slate: a
committed sha256-pinned topics XML for the three human fields, BeIR parquet corpus with the title
folded in, and BeIR tsv qrels with negative grades clamped. Each of those is a place a silent bug
would misalign the study, so each has a test.
"""

from __future__ import annotations

import pytest

from src.config import load_datasets
from src.data import load_dataset
from src.data.base import Dataset
from src.data.trec_covid import TrecCovid, _fold_title, _parse_topics


@pytest.fixture(scope="module")
def trec_covid() -> Dataset:
    return load_dataset("trec_covid")


def test_config_counts_and_pinned_qrel_provenance():
    cfg = load_datasets().dataset("trec_covid")
    assert (cfg.n_queries, cfg.n_docs, cfg.n_qrels) == (50, 171332, 66336)
    # qrels live in a separate BeIR repo, pinned; the corpus repo has no qrels of its own.
    assert cfg.qrels_hf_repo == "BeIR/trec-covid-qrels"
    assert cfg.qrels_hf_revision and cfg.hf_revision


def test_prose_tokenizer():
    tok = load_datasets().dataset("trec_covid").tokenizer
    assert (tok.stemming, tok.remove_stopwords) == (True, True)


def test_topics_xml_pins_three_distinct_fields_for_all_50_topics():
    """The sha256 assertion lives in _parse_topics; if the committed file drifts this raises. All
    50 topics must carry three genuinely different phrasings of the same need."""
    topics = _parse_topics()
    assert len(topics) == 50
    for fields in topics.values():
        assert set(fields) == {"query", "question", "narrative"}
        assert fields["query"] != fields["question"] != fields["narrative"]
        assert all(fields[f].strip() for f in fields)
    # anchor on the plan's worked example
    assert topics["1"]["query"] == "coronavirus origin"
    assert topics["1"]["question"] == "what is the origin of COVID-19"


def test_title_is_folded_into_text_and_emptied():
    """The dense build embeds `text` alone; BEIR encodes title+" "+text. So the title must move
    into `text` (matching BEIR) and `title` must end empty (matching the build's assumption)."""
    doc = _fold_title("Clinical features", "A retrospective review")
    assert doc == {"title": "", "text": "Clinical features A retrospective review"}
    # no title -> text unchanged, no stray leading separator
    assert _fold_title("", "body only") == {"title": "", "text": "body only"}
    assert _fold_title("   ", "body only") == {"title": "", "text": "body only"}


def test_negative_qrel_grades_are_clamped(tmp_path, monkeypatch):
    """trec_eval treats grade <= 0 as non-relevant; the two -1 judgments must become 0 so the grade
    set stays 0/1/2. 0-grade judged docs are kept (they are part of the judged pool)."""
    tsv = tmp_path / "test.tsv"
    tsv.write_text("query-id\tcorpus-id\tscore\n1\tA\t2\n1\tB\t-1\n1\tC\t0\n2\tD\t1\n")
    adapter = TrecCovid(load_datasets().dataset("trec_covid"))
    monkeypatch.setattr(adapter, "_hf_qrels", lambda: str(tsv))
    qrels = adapter._load_qrels()
    assert qrels == {"1": {"A": 2, "B": 0, "C": 0}, "2": {"D": 1}}


@pytest.mark.integration
def test_counts_and_qrels_resolve(trec_covid: Dataset):
    """Runs validate(), which also exercises the question<->BeIR merge identity assertion inside
    _load_queries (the topic-number to BeIR-id merge must be exact)."""
    trec_covid.validate()


@pytest.mark.integration
def test_all_three_human_fields_align_with_qrels(trec_covid: Dataset):
    assert isinstance(trec_covid, TrecCovid)
    fields = trec_covid.human_query_fields()
    qids = set(trec_covid.qrels())
    for name in ("query", "question", "narrative"):
        assert set(fields[name]) == qids, f"{name} field does not cover every judged topic"
    # narrative is the elaborated form -- reliably longer than the keyword query
    assert all(len(fields["narrative"][q]) > len(fields["query"][q]) for q in qids)


@pytest.mark.integration
def test_corpus_titles_folded_and_qrels_graded(trec_covid: Dataset):
    corpus, qrels = trec_covid.corpus(), trec_covid.qrels()
    assert all(d["title"] == "" for d in corpus.values()), "titles must be folded into text"
    grades = {s for rels in qrels.values() for s in rels.values()}
    assert grades == {0, 1, 2}, "TREC-COVID qrels are graded 0/1/2 (with -1 clamped to 0)"
