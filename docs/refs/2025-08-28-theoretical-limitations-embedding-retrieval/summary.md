# On the Theoretical Limitations of Embedding-Based Retrieval

|  |  |
|---|---|
| **Authors** | Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee (Google DeepMind / JHU) |
| **Venue / date** | arXiv:2508.21038 — v1 2025-08-28 |
| **Links** | [abs](https://arxiv.org/abs/2508.21038) · [pdf](https://arxiv.org/pdf/2508.21038) · local: `original.pdf` |
| **Type** | Preprint (also on OpenReview) |

## Summary

Connects **communication-complexity / sign-rank** results to single-vector embeddings, proving that
for any fixed embedding dimension *d* there exist sets of top-*k* document combinations that **no
dot-product embedding can return.** Introduces the **LIMIT** stress-test dataset, on which even SOTA
embedders fail despite the task's apparent simplicity.

## Relevance to this study

A **fundamental ceiling on what single-vector embedding leaderboards can measure at all.** It bears on
our story in two ways: (1) it is a first-principles reason that embedding-benchmark numbers can be
brittle or misleading — there are query/document relationships the architecture simply cannot
represent, so a leaderboard's ordering is contingent on which of those a benchmark happens to probe;
(2) it strengthens the general "the metric/model can be blind to real structure" theme that our
sparse-vs-dense finding instantiates empirically. Use it in the framing/limitations discussion as the
theoretical bookend to our empirical metric-blindness result.

## Cite for
- A provable capacity limit of single-vector (dot-product) embedding retrieval (sign-rank / dimension).
- First-principles reason embedding-leaderboard numbers can mislead — theoretical bookend to metric blindness.
