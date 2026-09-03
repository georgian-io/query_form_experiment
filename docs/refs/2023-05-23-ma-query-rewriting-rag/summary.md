# Query Rewriting for Retrieval-Augmented Large Language Models

|  |  |
|---|---|
| **Authors** | Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan |
| **Venue / date** | arXiv:2305.14283 — v1 2023-05-23 (EMNLP 2023) |
| **Links** | [abs](https://arxiv.org/abs/2305.14283) · [pdf](https://arxiv.org/pdf/2305.14283) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

Introduces the **Rewrite-Retrieve-Read** framework for RAG: instead of feeding the user's input
straight to the retriever, an **LLM first rewrites the query** to close "the gap between the input
text and the needed knowledge." A small trainable rewriter is tuned by reinforcement learning on the
black-box reader's feedback. Improves open-domain and multiple-choice QA.

## Relevance to this study

The **canonical primary source for our motivating premise** — that in production RAG the query
reaching the retriever is *LLM-authored*, not the user's raw words. When the reframed brief says
"increasingly the query is written by an LLM," this is the paper to cite. It establishes query
rewriting as a first-class RAG component and thereby makes our question — *does that rewriting
reshuffle which embedder wins?* — the natural next one.

Note the direction of their contribution vs ours: Ma et al. **optimize** the rewrite to improve a
downstream reader; we hold the retriever/corpus fixed and **measure** how rewrite *style* (terse /
verbose / paraphrase) moves the retrieval leaderboard. Their work is why query rewriting matters;
ours is what it does to evaluation.

## Cite for
- Primary evidence that LLMs rewrite queries inside production RAG pipelines (study motivation).
- The retrieve-then-read → rewrite-retrieve-read framing.
