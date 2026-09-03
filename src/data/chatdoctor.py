"""ChatDoctor_HealthCareMagic -- colloquial patient narrative -> doctor's answer (§6).

Standard RTEB packaging, so loading is inherited. What matters here is everything around it:

* **Real de-identified patient text.** Non-negotiable §7: no raw commits, region-appropriate
  storage. The normalized parquet cache lands under the git-ignored `.cache/`, and the corpus
  is written to turbopuffer in whatever region `TURBOPUFFER_REGION` names -- choose that
  deliberately rather than by default.

* **The corpus is 5,545 docs, not the 112k the plan's §6 table quotes.** RTEB packaged a subset
  of HealthCareMagic. That single number is why this dataset was scheduled last as the expensive
  one; it is in fact cheap. 5,591 queries, exactly one relevant document each.

* **`_id`, not `id`.** Alone among the three datasets we load, this one follows mteb's field
  name. The base loader accepts either.

* **Highest unlabeled-relevant risk on the slate (§5.6).** The "documents" are doctors' answers
  to patient questions, and many different patients ask near-identical questions -- so several
  answers in the corpus may genuinely serve a query while only one is labelled. Published
  `recall@100` is 0.9619 rather than 1.0, confirming this is a real retrieval task over a corpus
  larger than top-k, but a measured ΔNDCG here is more likely than elsewhere to reflect which
  *equally good* answer surfaced rather than a true relevance change. That is exactly what the
  §5.6 audit's unlabeled-relevant check exists to quantify.
"""

from __future__ import annotations

from src.data.rteb import RTEBJsonlDataset


class ChatDoctorHealthCareMagic(RTEBJsonlDataset):
    name = "chatdoctor"
