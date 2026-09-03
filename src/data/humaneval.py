"""HumanEval -- function docstring/spec -> the Python implementation that satisfies it (§6).

Standard RTEB packaging with no loading quirks; what matters here is downstream of the adapter:

  * **Code tokenizer.** `datasets.yaml` turns off stemming and stopword removal and turns on
    case sensitivity for this dataset. Stemming mangles identifiers (`enumerate` -> `enumer`),
    stopword removal eats keywords like `in`, `not`, `for`, and case folding collapses the
    `numbers`/`Numbers` distinction that code actually depends on (§5.3).
  * **Highest qrel risk on the slate.** The query *is* the specification, so a transform that
    rewrites the docstring can legitimately change which implementation satisfies it. §6 marks
    HyDE as out of bounds here, and paraphrase must be spec-preserving.
  * Relevance is genuinely 1:1 -- 158 queries, 158 solutions, one gold each. Verified against
    RTEB's published metrics rather than assumed: precision@100 = 0.01 = 1/100 and
    recall@100 = 1.0, which is what a single relevant document per query implies.

Note the packaged set is 158 problems, not the 164 of upstream HumanEval (the plan's §6 table
quotes the upstream figure).
"""

from __future__ import annotations

from src.data.rteb import RTEBJsonlDataset


class HumanEval(RTEBJsonlDataset):
    name = "humaneval"
