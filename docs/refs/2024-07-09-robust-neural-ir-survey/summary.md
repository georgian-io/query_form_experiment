# Robust Neural Information Retrieval: An Adversarial and Out-of-Distribution Perspective

|  |  |
|---|---|
| **Authors** | Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Xueqi Cheng, et al. |
| **Venue / date** | arXiv:2407.06992 — v1 2024-07-09 (survey; introduces the "BestIR" benchmark) |
| **Links** | [abs](https://arxiv.org/abs/2407.06992) · [pdf](https://arxiv.org/pdf/2407.06992) · local: `original.pdf` |
| **Type** | Preprint (survey) · companion SIGIR 2024 tutorial "Robust Information Retrieval" (arXiv:2406.08891) |

## Summary

Comprehensive survey organizing neural-IR robustness into two families: **adversarial robustness**
(imperceptible malicious perturbations) and **out-of-distribution (OOD) robustness** (query
variations, unseen queries/tasks, corpus shift). Taxonomizes attacks, defenses, and evaluation
benchmarks across dense retrievers and neural rankers.

## Relevance to this study

The **map that places our work in the field's vocabulary.** Our query-form study is, in this
taxonomy, an **OOD-query-variation robustness** study — specifically of the *meaning-preserving*
variety. Citing this survey lets the related-work section name the category cleanly and distinguish
our contribution: prior OOD-robustness work measures a single system's effectiveness drop; we measure
**cross-model leaderboard reshuffle** and add the **relevance-label-density (metric-blindness)** axis
the survey's evaluation discussion does not isolate.

Also folds in the SIGIR 2024 tutorial (arXiv:2406.08891) by the same group — cite either as the
canonical robustness-taxonomy reference.

## Cite for
- The standard adversarial-vs-OOD robustness taxonomy (positions our study as OOD query variation).
- A single citation covering the robustness literature we extend from per-system to cross-leaderboard.
