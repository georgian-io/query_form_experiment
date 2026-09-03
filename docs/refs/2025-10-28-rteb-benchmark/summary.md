# RTEB: A Real Test of How Well Models Actually Retrieve

|  |  |
|---|---|
| **Author / outlet** | mrlcf (Substack) — third-party explainer of RTEB |
| **Date** | 2025-10-28 |
| **Links** | [article](https://mrlcftech.substack.com/p/rteb-a-real-test-of-how-well-models) · [MTEB docs](https://docs.mteb.org/) · [MTEB/RTEB repo](https://github.com/embeddings-benchmark/mteb) · local: `original.html` |
| **Type** | ⚠️ **Third-party blog explainer — not the primary benchmark spec** (no arXiv paper for RTEB yet) |

## Summary

RTEB (Retrieval Embedding Benchmark) is the **retrieval-focused successor to MTEB** and **the
benchmark this project reproduces and perturbs.** Where MTEB spreads over 8 tasks, RTEB targets
retrieval with a unified NDCG@10 metric across production domains (legal, finance, code, medical,
scientific, multilingual news). Its signature design choice: **combine public and private/closed
datasets** so that a sharp public-vs-private gap flags contamination/leaderboard overfitting — "true
generalization, not memorization." Even strong models sit around mid-0.60s NDCG@10.

## Relevance to this study

The **object of study.** Our reproduction gate targets RTEB's published NDCG@10, our leaderboard is
the RTEB board, and our τ/RBO reshuffle is measured against it. This reference is here to (a) cite
what RTEB is and why it was built (contamination resistance, retrieval focus), and (b) explain why we
work with the datasets we do (its public component: AILAStatutes, HumanEval, CUREv1, etc.).

**Caveat / gap.** This is a third-party blog, not an authoritative spec — there is no arXiv/paper for
RTEB as of this writing. Treat it as orientation; cite the **MTEB paper**
([2022-10-13](../2022-10-13-mteb/summary.md)) for benchmark methodology and the MTEB
docs/leaderboard for the current RTEB dataset list and numbers. Replace this with the primary RTEB
writeup if/when one is published.

## Cite for
- What RTEB is: retrieval-focused, public+private, contamination-resistant, enterprise domains.
- Context for our reproduction gate and dataset selection (orientation only — not a primary source).
