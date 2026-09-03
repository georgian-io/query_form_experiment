# Promptagator: Few-shot Dense Retrieval From 8 Examples

|  |  |
|---|---|
| **Authors** | Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, Ming-Wei Chang (Google) |
| **Venue / date** | arXiv:2209.11755 — v1 2022-09-23 (ICLR 2023) |
| **Links** | [abs](https://arxiv.org/abs/2209.11755) · [pdf](https://arxiv.org/pdf/2209.11755) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

Promptagator uses an LLM as a **few-shot query generator** (≤8 task examples) to synthesize
task-specific training queries from documents, then trains a dual encoder + reranker on them.
Crucially, the few-shot examples make the generated queries **match the target task's query
distribution** rather than a generic one. The trained retrievers beat ColBERT-v2 by >1.2 nDCG on
average across 11 BEIR datasets.

## Relevance to this study

Speaks directly to **Q3 — are synthetic queries a fair proxy for human ones?** Promptagator's whole
design premise is that synthetic queries only resemble real ones **when explicitly conditioned to
match the target distribution** — otherwise LLM-generated queries drift toward generic phrasing.
That is the training-side statement of exactly the caveat our Q3 finds on the evaluation side: our
synthetic **terse** proxy matched human keywords well (validated), but synthetic **verbose** was a
weaker proxy for human narratives.

Distinction to note: Promptagator generates queries **from documents to train** retrievers; we
**transform existing queries to evaluate** leaderboards. Different direction, same underlying
question — how faithfully do LLM-authored queries stand in for human ones.

## Cite for
- Precedent that LLM synthetic queries need distribution-matching to resemble real user queries.
- Support for the Q3 caveat that synthetic-query conclusions require validation against human queries.
