"""TREC-COVID (BEIR) -- scientific-literature retrieval over the CORD-19 corpus (§1, §5.5).

Not an RTEB dataset, so it does not use RTEB's three-file jsonl packaging and cannot be gated
against an RTEB number (see the plan §4 deviation). It earns its place for two things no RTEB
dataset offers:

1. **Three human-authored phrasings of the same information need**, judged against the same qrels.
   Every TREC-COVID topic ships a keyword `query` ("coronavirus origin"), a one-sentence
   `question` ("what is the origin of COVID-19"), and a ~25-word `narrative`. These map onto our
   terse / baseline / verbose conditions with *zero intent drift by construction* -- the §5.6
   audit exists because our LLM rewrites change the information need 18-31% of the time; here there
   is nothing to audit. `question` is the baseline (the field BEIR scores, and the gate anchor).

2. **A methodology-validation layer**: run the LLM transforms on `question` and compare the
   synthetic terse/verbose against the human `query`/`narrative` for the same need.

Provenance, and how each piece is pinned (§8 item 5):

* **Corpus + qrels: BeIR HF repos, pinned by revision.** Corpus is `BeIR/trec-covid` (171,332
  CORD-19 title+abstract docs, the 2020-07-16 snapshot BeIR packaged); qrels are the Round-5
  complete judgments in `BeIR/trec-covid-qrels/test.tsv` (graded 0/1/2), restricted to the docs
  BeIR ships. Both revisions are in datasets.yaml.
* **The three fields: `topics-rnd5.xml`, committed and content-pinned.** NIST's topics file is not
  a versioned repo, so rather than fetch it at load time (the "rely on a moving endpoint" trap §9
  forbids) it is committed under `resources/` and its sha256 asserted on every load. Its `question`
  field is verified equal to BeIR's own query text for all 50 topics -- that identity is what makes
  the topic-number <-> BeIR-id merge sound.

Two reproduction details that bite only here, both handled below:

* **Real titles.** This is the first dataset on the slate whose corpus has non-empty titles. The
  dense build embeds `corpus[doc]["text"]` alone (`01_build_index.py`), and BEIR encodes
  `title + " " + text`, so the title is folded into `text` here (leaving `title` empty) exactly as
  AILAStatutes folds its statute name in. Without this the dense gate would diverge badly.
* **Negative qrel grades.** Two judgments carry score -1; trec_eval treats any grade <= 0 as
  non-relevant, so they are clamped to 0 (standard BEIR practice) to keep the grade set 0/1/2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree.ElementTree import Element

import pandas as pd
from defusedxml.ElementTree import fromstring as _xml_fromstring
from huggingface_hub import hf_hub_download

from src.data.base import Corpus, Dataset, Document, Qrels, Queries

_TOPICS_XML = Path(__file__).resolve().parent / "resources" / "trec_covid" / "topics-rnd5.xml"
# Round-5 topics, fetched from ir.nist.gov/covidSubmit/data/topics-rnd5.xml and committed. Pinned
# by content hash because the source is an unversioned static file, not a repo revision.
_TOPICS_SHA256 = "4fc339ae8333a545ca50826357adf5eec8434df557bbce2dc40e8efd01380f42"

_FIELDS = ("query", "question", "narrative")
_BASELINE_FIELD = "question"

# BeIR ships corpus/queries as a single parquet shard each, at these paths within the pinned
# revision. Written out rather than globbed so a repackaging upstream fails loudly here.
_CORPUS_PARQUET = "corpus/corpus-00000-of-00001.parquet"
_QUERIES_PARQUET = "queries/queries-00000-of-00001.parquet"


class TrecCovid(Dataset):
    name = "trec_covid"

    def _load_corpus(self) -> Corpus:
        df = pd.read_parquet(self._hf(_CORPUS_PARQUET))
        return {
            str(row["_id"]): _fold_title(str(row["title"]), str(row["text"]))
            for row in df.to_dict("records")
        }

    def _load_queries(self) -> Queries:
        """The baseline `question` field, verified against BeIR's own query text so the merge is
        sound."""
        fields = self.human_query_fields()
        question = fields[_BASELINE_FIELD]
        beir = pd.read_parquet(self._hf(_QUERIES_PARQUET))
        beir_q = {str(r["_id"]): str(r["text"]) for r in beir.to_dict("records")}
        if beir_q != question:
            raise ValueError(
                f"{self.name}: topics-rnd5.xml `question` field does not match BeIR's query text; "
                "the topic-number <-> BeIR-id merge cannot be trusted"
            )
        return question

    def _load_qrels(self) -> Qrels:
        tsv = pd.read_csv(
            self._hf_qrels(),
            sep="\t",
            dtype={"query-id": str, "corpus-id": str, "score": int},
        )
        qrels: Qrels = {}
        for row in tsv.to_dict("records"):
            # trec_eval treats grade <= 0 as non-relevant; clamp the two -1 judgments to 0 so the
            # grade set stays 0/1/2 (standard BEIR practice). NDCG@10 is unchanged either way.
            score = max(int(row["score"]), 0)
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = score
        return qrels

    def human_query_fields(self) -> dict[str, Queries]:
        """All three human-authored phrasings per topic (§1), keyed by field name.

        `question` is the baseline that `queries()` returns; `query` (keyword, ~terse) and
        `narrative` (elaborated, ~verbose) are the drift-free alternative conditions. Fed straight
        to retrieve, bypassing the transform/generator layer -- there is no generation.
        """
        topics = _parse_topics()
        return {field: {num: fields[field] for num, fields in topics.items()} for field in _FIELDS}

    def _hf(self, filename: str) -> str:
        return hf_hub_download(
            repo_id=self.config.hf_repo,
            filename=filename,
            revision=self.config.hf_revision,
            repo_type="dataset",
        )

    def _hf_qrels(self) -> str:
        return hf_hub_download(
            repo_id=self.config.qrels_hf_repo,
            filename=self.config.qrels_path,
            revision=self.config.qrels_hf_revision,
            repo_type="dataset",
        )


def _fold_title(title: str, text: str) -> Document:
    """Fold the title into `text` (BEIR encodes `title + " " + text`), leaving `title` empty so the
    dense build -- which embeds `text` alone -- matches BEIR's document encoding."""
    joined = f"{title} {text}".strip() if title.strip() else text.strip()
    return Document(title="", text=joined)


def _parse_topics() -> dict[str, dict[str, str]]:
    raw = _TOPICS_XML.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != _TOPICS_SHA256:
        raise ValueError(
            f"{_TOPICS_XML.name}: sha256 {actual} != pinned {_TOPICS_SHA256}; the topics file has "
            "changed and the three human fields can no longer be trusted"
        )
    topics: dict[str, dict[str, str]] = {}
    for topic in _xml_fromstring(raw).findall("topic"):
        num = topic.attrib["number"]
        topics[num] = {field: _text(topic, field) for field in _FIELDS}
    return topics


def _text(topic: Element, tag: str) -> str:
    element = topic.find(tag)
    if element is None or not element.text:
        raise ValueError(f"topic {topic.attrib.get('number')} missing <{tag}>")
    return element.text.strip()
