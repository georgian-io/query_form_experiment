"""AILAStatutes -- Indian Supreme Court case scenarios -> relevant statutes (§6).

Standard RTEB packaging, so the loading itself is inherited. The one quirk worth recording is
where the qrels come from, and it is expressed in `datasets.yaml` rather than in code:

**RTEB's published `relevance.jsonl` is truncated.** It holds 50 rows -- the first relevant
statute per query -- but the leaderboard was scored against all 217 (2-5 per query). RTEB's own
published metrics prove it two ways:

  * The 82-doc corpus is smaller than RTEB's top_k of 100, so every document is retrieved for
    every query and precision@100 reduces to mean(relevant per query)/100. The published value
    is 0.0434; 217/50/100 = 0.0434 exactly, while the truncated file predicts 0.0100.
  * recall@1 (0.135) differs from precision@1 (0.54), which is impossible if each query has
    exactly one relevant document.

`mteb/AILA_statutes` carries those 217 pairs alongside a byte-identical corpus and query set, so
config points qrels there and the base class re-verifies that identity on every load.

The other trap: the dataset card's prose says 197 statutes; the packaged corpus is 82. Trust the
pinned file.
"""

from __future__ import annotations

from src.data.rteb import RTEBJsonlDataset


class AILAStatutes(RTEBJsonlDataset):
    name = "aila_statutes"
