# How You Ask Matters! Adaptive RAG Robustness to Query Variations

|  |  |
|---|---|
| **Authors** | Yunah Jang, Megha Sundriyal, Kyomin Jung, Meeyoung Cha |
| **Venue / date** | arXiv:2604.10745 — v1 2026-04-12 |
| **Links** | [abs](https://arxiv.org/abs/2604.10745) · [pdf](https://arxiv.org/pdf/2604.10745) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Builds a benchmark of **semantically identical query variations** (human-written and model-generated
rewrites) to stress-test Adaptive RAG. Finds a critical robustness gap: small, meaning-preserving
surface changes **dramatically alter retrieval decisions and answer accuracy**, and larger models do
**not** close the gap.

## Relevance to this study

Almost our exact experimental frame, one layer downstream. They hold intent fixed, vary surface form,
and measure disruption — but at the level of an end-to-end RAG system's retrieval decisions and
answers, where we measure it as **leaderboard reshuffle (τ/RBO) across embedding models**. Their
"robustness gap" is the RAG-side analogue of our reshuffle. Two shared conclusions strengthen ours:
disruption is real for *meaning-preserving* rewrites (not just typos/adversarial), and **scale does
not fix it** — mirroring our finding that the strongest models are not immune.

## Cite for
- Contemporary evidence that RAG retrieval is brittle to meaning-preserving query variation.
- Scale does not close the query-variation robustness gap (parallel to our per-model result).
