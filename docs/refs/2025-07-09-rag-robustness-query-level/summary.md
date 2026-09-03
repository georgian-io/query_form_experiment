# Investigating the Robustness of Retrieval-Augmented Generation at the Query Level

|  |  |
|---|---|
| **Authors** | Sezen Perçin, Xin Su, Qutub Sha Syed, Phillip Howard, Aleksei Kuvshinov, Leo Schwinn, Kay-Ulrich Scholl |
| **Venue / date** | arXiv:2507.06956 — v1 2025-07-09 (GEM Workshop @ ACL 2025) |
| **Links** | [abs](https://arxiv.org/abs/2507.06956) · [pdf](https://arxiv.org/pdf/2507.06956) · local: `original.pdf` |
| **Type** | Peer-reviewed (ACL 2025 workshop) |

## Summary

Runs **1,092 experiments** perturbing queries across general and domain-specific datasets to isolate
where RAG breaks. Finds retriever performance can **degrade significantly even under minor query
variations**, and offers an evaluation framework plus recommendations.

## Relevance to this study

A **peer-reviewed, component-level** anchor for query-form sensitivity in the retriever specifically
(not just end-to-end RAG). Where our contribution is the cross-model leaderboard view, this gives a
controlled single-pipeline magnitude baseline for "how much does reformulation move the retriever,"
useful for calibrating our per-model ΔNDCG and for citing that the effect is established at the
component level, not only in our leaderboard aggregate.

## Cite for
- Controlled evidence that retriever effectiveness degrades under minor query variation.
- A component-level (retriever-isolated) methodology and magnitude baseline for query-form effects.
