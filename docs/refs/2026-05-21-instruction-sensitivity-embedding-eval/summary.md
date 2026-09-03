# One Prompt is Not Enough: Instruction Sensitivity Undermines Embedding Model Evaluation

|  |  |
|---|---|
| **Authors** | Yevhen Kostiuk, Kenneth Enevoldsen |
| **Venue / date** | arXiv:2605.22544 — v1 2026-05-21 |
| **Links** | [abs](https://arxiv.org/abs/2605.22544) · [pdf](https://arxiv.org/pdf/2605.22544) · local: `original.pdf` |
| **Type** | Preprint (Enevoldsen is an MTEB maintainer) |

## Summary

Empirical study over **6 embedding models × 11 datasets × 15 task prompts (990 evaluations)**:
instruction-tuned embedding models are **highly sensitive to prompt phrasing**, and the single
default prompt used in benchmarks can systematically over- or under-state performance. The headline:
the **leaderboard ranking is not robust to prompt selection — by choosing prompts favorably, any
model in the study can be promoted to first place.**

## Relevance to this study

The **closest published analog to our thesis on the instruction axis.** It measures embedding-
leaderboard reshuffling driven by *input/instruction phrasing* and concludes single-prompt evaluation
is unreliable — the same shape as our τ/RBO leaderboard-stability result, expressed through the
instruction knob instead of the query-content knob.

Crucially, it **validates a core design choice of ours.** Our CLAUDE.md rule holds the *query
instruction constant across transform conditions* precisely so ΔNDCG measures query *content*, not
instruction change. This paper is the evidence that had we let the instruction vary, it would have
confounded everything — models can be moved to #1 by prompt choice alone. Cite it both as convergent
evidence (leaderboards move under phrasing) and as the justification for our instruction-pinning
discipline. The two axes are complementary: they vary the instruction and hold content fixed; we vary
content and hold the instruction fixed.

## Cite for
- Embedding leaderboards are not robust to prompt/instruction choice (any model → #1 by prompt selection).
- Direct justification for holding the query instruction constant across our transform conditions.
