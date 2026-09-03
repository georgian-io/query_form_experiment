# On the Robustness of LLM-Based Dense Retrievers: A Systematic Analysis of Generalizability and Stability

|  |  |
|---|---|
| **Authors** | Yongkang Li, Panagiotis Eustratiadis, Yixing Fan, Evangelos Kanoulas |
| **Venue / date** | arXiv:2604.16576 — v1 2026-04-17 |
| **Links** | [abs](https://arxiv.org/abs/2604.16576) · [pdf](https://arxiv.org/pdf/2604.16576) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Systematically evaluates **LLM-backbone dense retrievers** across 30 datasets on generalizability and
stability. Key split: they **resist typos and corpus poisoning** but remain **vulnerable to semantic
(meaning-level) perturbations**, and reasoning-optimized models generalize *worse*.

## Relevance to this study

The most model-matched paper in the collection: our leaderboard is topped by exactly this class —
`qwen3-embedding-8b`, `gemini-embedding-001` are LLM-backbone embedders. The **typo-robust /
semantics-fragile** finding is a mechanism-level prediction of our results: our *surface-ish* move
(terse keywording) should reshuffle less than our *semantic* moves (paraphrase, and especially
verbose broadening), because these models are precisely fragile to meaning-level change. Their
result also cautions that "stronger/reasoning-tuned" ≠ "more robust," consistent with our per-model
reshuffle not tracking raw quality.

## Cite for
- LLM-backbone dense retrievers are typo-robust but semantics-fragile (predicts which of our conditions reshuffle).
- Robustness is model-class- and training-dependent, not monotone in capability.
