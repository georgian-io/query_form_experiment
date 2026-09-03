# Dense Retrievers Can Fail on Simple Queries: Revealing the Granularity Dilemma of Embeddings

|  |  |
|---|---|
| **Authors** | Liyan Xu, Zhenlin Su, Mo Yu, et al. |
| **Venue / date** | arXiv:2506.08592 — v1 2025-06-10 (EMNLP 2025 Findings) |
| **Links** | [abs](https://arxiv.org/abs/2506.08592) · [pdf](https://arxiv.org/pdf/2506.08592) · local: `original.pdf` |
| **Type** | Peer-reviewed (EMNLP 2025 Findings) |

## Summary

Introduces **CapRetrieval** (3,024 passages, 404 queries, ~1.3M **densely annotated** query–passage
pairs with graded 0/1/2 relevance) to show dense encoders fail on fine-grained entity/event matching
**regardless of model size or training source** — a "granularity dilemma." A targeted fine-tuned
0.1B encoder beats a 7B SOTA model on the task.

## Relevance to this study

Two hooks, both central:

1. **Dense-annotation methodology mirrors ours.** Its ~1.3M graded 0/1/2 judgments are built exactly
   so the benchmark can *see* fine-grained failures — the same reason our **dense** datasets
   (CUREv1 ~40 golds, TREC-COVID ~494 graded golds) register the verbose-elaboration damage a
   **sparse** 1-gold benchmark cannot. Independent support for the metric-blindness argument: coarse
   qrels hide failures that dense qrels expose.
2. **Failure is not size-monotone.** Small targeted models beat a 7B — echoing our per-model reshuffle
   that does not track raw model scale/quality.

## Cite for
- Densely-graded qrels reveal retrieval failures invisible to coarse/sparse benchmarks (metric-blindness support).
- Dense-retrieval failure is not monotone in model size.
