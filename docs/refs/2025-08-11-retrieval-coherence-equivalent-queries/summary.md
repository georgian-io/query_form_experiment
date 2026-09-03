# Improving Document Retrieval Coherence for Semantically Equivalent Queries

|  |  |
|---|---|
| **Authors** | Stefano Campese, Alessandro Moschitti, Ivano Lauriola |
| **Venue / date** | arXiv:2508.07975 — v1 2025-08-11 |
| **Links** | [abs](https://arxiv.org/abs/2508.07975) · [pdf](https://arxiv.org/pdf/2508.07975) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Shows dense retrievers return **divergent top-k documents for semantically equivalent but lexically
varied queries**, then proposes a modified Multi-Negative Ranking loss that penalizes top-k
discrepancy across paraphrases. Reports lower sensitivity plus accuracy gains on MS MARCO, NQ, BEIR,
and TREC DL.

## Relevance to this study

Quantifies our exact phenomenon — **ranking instability under meaning-preserving rewording** — and,
tellingly, frames it as **top-k overlap divergence**, which is conceptually our RBO/τ measured on a
single query's document ranking rather than on the model leaderboard. It is the per-query,
single-model view of what we aggregate into a cross-model leaderboard reshuffle. Its mitigation
(train paraphrase-invariance in) is the natural "so what do we do about it" companion to our
measurement, and worth citing in a remedies/implications paragraph.

## Cite for
- Dense retrievers give divergent top-k results for paraphrased/equivalent queries (top-k overlap).
- A training-time remedy (paraphrase-invariant contrastive objective) for query-form sensitivity.
