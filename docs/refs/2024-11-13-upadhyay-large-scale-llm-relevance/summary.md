# A Large-Scale Study of Relevance Assessments with Large Language Models: An Initial Look

|  |  |
|---|---|
| **Authors** | Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, Ian Soboroff, Hoa Trang Dang, Jimmy Lin |
| **Venue / date** | arXiv:2411.08275 — v1 2024-11-13 (ICTIR 2025) |
| **Links** | [abs](https://arxiv.org/abs/2411.08275) · [pdf](https://arxiv.org/pdf/2411.08275) · local: `original.pdf` |
| **Type** | Peer-reviewed (ICTIR 2025) |

## Summary

Deploys **UMBRELA** LLM relevance judgments across **77 runs / 19 teams** in the TREC 2024 RAG Track,
comparing fully manual vs three LLM-assisted strategies. Finds that **run-level system rankings from
LLM judgments correlate strongly with manual ones — Kendall τ ≈ 0.89** on nDCG@20/@100 and Recall@100
— though **per-topic agreement is weaker.**

## Relevance to this study

The central "LLM judgments preserve the leaderboard" evidence, and it speaks our language: it reports
its result as a **Kendall τ between rankings** (~0.89), exactly our metric. Two uses:

1. **Licenses the audit's premise** — an LLM can stand in for human relevance judgment well enough
   that *system rankings* are largely preserved, which is what our drift audit implicitly relies on.
2. **Calibrates our threshold intuition** — even a broadly-trusted LLM-judgment substitution lands
   around τ ≈ 0.89, i.e. *below* our 0.9 go/no-go bar. That is a useful yardstick: it shows τ in the
   high-0.8s is a meaningful, non-trivial amount of reshuffle, reinforcing that our sub-0.9 query-form
   effects are real, not noise.

The "weaker per-topic agreement" caveat also parallels the [Shelf-Life](../2025-02-28-shelf-life-test-collections/summary.md)
"aggregate stable, individual sensitive" nuance.

## Cite for
- LLM relevance judgments preserve *system rankings* at aggregate (τ ≈ 0.89), weaker per-topic.
- A τ yardstick: high-0.8s reshuffle is meaningful — supports reading our sub-0.9 effects as real.
