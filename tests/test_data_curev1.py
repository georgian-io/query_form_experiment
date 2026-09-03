"""§5.1 correctness gate for the CUREv1_en adapter.

The load-bearing tests here are (1) the custom `{qid: {docid: score}}` qrel parsing, which no
other RTEB dataset uses, and (2) that the qrels are genuinely dense (~40 relevant/query) -- the
one property that makes CUREv1 the metric-blindness counterweight to ChatDoctor's single gold.
"""

from __future__ import annotations

import statistics

import pytest

from src.config import load_datasets
from src.data import load_dataset
from src.data.base import Dataset
from src.data.curev1 import CUREv1En


@pytest.fixture(scope="module")
def curev1() -> Dataset:
    return load_dataset("curev1_en")


def test_config_counts_are_the_packaged_figures():
    cfg = load_datasets().dataset("curev1_en")
    assert (cfg.n_queries, cfg.n_docs, cfg.n_qrels) == (2000, 244600, 80716)


def test_prose_tokenizer_not_the_code_one():
    """Biomedical passages are prose: stemming and stopword removal both apply."""
    tok = load_datasets().dataset("curev1_en").tokenizer
    assert (tok.stemming, tok.remove_stopwords, tok.case_sensitive) == (True, True, False)


def test_qrels_parse_the_dict_per_line_format_not_beir_triples():
    """CUREv1 packs qrels as one `{qid: {docid: score}}` object per line, unlike every other RTEB
    dataset. Getting this wrong would silently produce empty qrels."""
    cfg = load_datasets().dataset("curev1_en")
    adapter = CUREv1En(cfg)
    adapter._read_jsonl = lambda filename: [  # type: ignore[method-assign]
        {"q1": {"d1": 1, "d2": 1}},
        {"q2": {"d3": 1}},
    ]
    qrels = adapter._load_qrels()
    assert qrels == {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}}


@pytest.mark.integration
def test_counts_and_qrels_resolve(curev1: Dataset):
    curev1.validate()


@pytest.mark.integration
def test_qrels_are_dense_not_single_gold(curev1: Dataset):
    """The reason CUREv1 exists: unlike ChatDoctor's exactly-one-gold, NDCG@10 must see a whole
    top-10 of relevant passages. Avg ~40 relevant/query, and it is emphatically not 1:1."""
    qrels = curev1.qrels()
    counts = [len(v) for v in qrels.values()]
    assert min(counts) >= 1
    assert statistics.mean(counts) > 30  # ~40.4 in the packaged revision
    assert {1} != set(counts)  # not the ChatDoctor single-gold degeneracy
