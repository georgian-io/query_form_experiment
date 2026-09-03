"""CUREv1_en -- expert clinical questions -> relevant medical passages (§2, plan-trec-covid).

The metric-blindness counterweight to ChatDoctor: same medical domain, structural opposite. Where
ChatDoctor is 5,545 docs with exactly one gold per query -- so NDCG@10 degenerates to "where did
the single gold land" -- CUREv1 is 244,600 passages with an average of 40.4 relevant passages per
query (min 1, max 1,364), so NDCG@10 sees the whole top-10 composition. The ChatDoctor<->CUREv1
pair is close to a designed experiment on whether verbose's "harmless broadening" is real
robustness or one-gold metric blindness.

Two quirks, both absorbed here:

* **`relevance.jsonl` is NOT BEIR triples.** Unlike every other RTEB dataset we load, CUREv1 packs
  qrels as one JSON object per line, `{qid: {docid: score, ...}}`, so the inherited
  `RTEBJsonlDataset._load_qrels` (which expects `query-id`/`corpus-id`/`score` rows) does not
  apply. That is the whole reason this adapter exists rather than being a bare `name = ...`
  subclass. Corpus and queries use the standard `id`/`text` shape, so those are inherited.

* **RTEB packages CUREv1_en as BINARY, not graded.** The dataset card describes graded judgments
  (Relevant / Partially Relevant / Not Relevant), but the packaged `relevance.jsonl` lists only
  the relevant pairs, every one at score 1 -- the "Not Relevant" pairs are simply absent and no
  "Partially Relevant" grade survives. This corrects the plan's "graded (Rel/Partial/Not)"
  framing. It does not weaken the metric-blindness contrast: the load-bearing property is ~40
  relevant passages per query (dense), which holds, so NDCG@10 still sees the whole top-10. We
  score against exactly this packaging because that is what the RTEB leaderboard number used.

Corpus rows carry no `title` (the base loader defaults it to ""), and the dense build embeds the
passage text alone -- matching how RTEB scored it.
"""

from __future__ import annotations

from src.data.base import Qrels
from src.data.rteb import RTEBJsonlDataset


class CUREv1En(RTEBJsonlDataset):
    name = "curev1_en"

    def _load_qrels(self) -> Qrels:
        """One object per line: {qid: {docid: score}}. See the module docstring for why this is
        not the inherited BEIR-triple format."""
        qrels: Qrels = {}
        for row in self._read_jsonl("relevance.jsonl"):
            (qid, rels), = row.items()
            qrels[qid] = {docid: int(score) for docid, score in rels.items()}
        return qrels
