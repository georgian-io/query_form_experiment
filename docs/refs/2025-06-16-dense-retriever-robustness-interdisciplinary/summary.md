# Evaluating the Robustness of Dense Retrievers in Interdisciplinary Domains

|  |  |
|---|---|
| **Authors** | Sarthak Chaturvedi, Anurag Acharya, Rounak Meyur, Koby Hayashi, Sai Munikoti, Sameera Horawalavithana |
| **Venue / date** | arXiv:2506.21581 — v1 2025-06-16 |
| **Links** | [abs](https://arxiv.org/abs/2506.21581) · [pdf](https://arxiv.org/pdf/2506.21581) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Shows that **benchmark construction** — whether topics/labels are cleanly separated vs semantically
overlapping — drives whether domain adaptation *appears* to help dense retrievers at all. The same
models look very different depending on the evaluation methodology, so measured robustness is partly
an **artifact of how the benchmark is labeled.**

## Relevance to this study

Independent, direct support for **Capstone A (metric blindness)** from a different angle. Our claim is
that a benchmark's relevance-label *density* governs what disruption it can register (sparse
1-gold ChatDoctor hides verbose's damage; dense CUREv1/TREC-COVID reveal it). This paper makes the
adjacent claim that label *structure/separation* governs what a benchmark can register about
robustness/adaptation. Same thesis family — **what you can measure depends on how the qrels were
built, not only on the model** — which is exactly the point we want corroborated by someone else.

## Cite for
- Measured retriever robustness/adaptation is partly an artifact of benchmark labeling/construction.
- External support that qrel structure — not just the model — governs what a benchmark can detect.
