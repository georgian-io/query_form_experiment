# MTEB: Massive Text Embedding Benchmark

|  |  |
|---|---|
| **Authors** | Niklas Muennighoff, Nouamane Tazi, Loïc Magne, Nils Reimers |
| **Venue / date** | arXiv:2210.07316 — v1 2022-10-13 (EACL 2023) |
| **Links** | [abs](https://arxiv.org/abs/2210.07316) · [pdf](https://arxiv.org/pdf/2210.07316) · local: `original.pdf` |
| **Type** | Peer-reviewed benchmark paper |

## Summary

MTEB is the standard embedding leaderboard: 8 tasks, 58 datasets, 112 languages, 33 models
benchmarked at release, with open code and a public leaderboard. Headline finding: **no single
embedding method dominates across tasks** — the field has not converged on a universal embedder.

## Relevance to this study

The **leaderboard-construction framework our study operates inside.** RTEB (the benchmark we
reproduce and perturb) is a retrieval-focused descendant of MTEB, so MTEB is the right primary
citation for how these leaderboards are built and read.

Framing hooks:
- "No method dominates across tasks" is the static-axis version of our finding: **which embedder
  ranks best is contingent on the evaluation setup.** We add a *new* axis of contingency they hold
  fixed — the **phrasing of the query** — and show the ranking moves along it too.
- MTEB fixes queries as given; our whole contribution is to vary that one held-constant factor and
  measure leaderboard stability (τ/RBO). Good "they held X fixed; we vary it" hinge.

## Cite for
- How embedding leaderboards are constructed and interpreted (RTEB's lineage).
- The premise that best-model rank is contingent on evaluation design.
