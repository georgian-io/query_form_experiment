# LLM-based Relevance Assessment Still Can't Replace Human Relevance Assessment

|  |  |
|---|---|
| **Authors** | Charles L. A. Clarke, Laura Dietz |
| **Venue / date** | arXiv:2412.17156 — v1 2024-12-22 |
| **Links** | [abs](https://arxiv.org/abs/2412.17156) · [pdf](https://arxiv.org/pdf/2412.17156) · local: `original.pdf` |
| **Type** | Preprint (position / critique) |

## Summary

The direct rebuttal to "LLMs can replace human relevance judges." Argues that despite high **aggregate
rank correlation**, LLM judgments can **distort system rankings** — especially through **circularity**
when the systems under evaluation share architecture/training with the judge — and that a high Kendall
τ **masks failure modes** rather than proving equivalence.

## Relevance to this study

The **essential counterweight** to [Upadhyay et al.](../2024-11-13-upadhyay-large-scale-llm-relevance/summary.md),
and the paper that most sharply justifies a specific choice in our **§5.6 drift audit**: we run the
judge from a **different model family than the rewriter** precisely to avoid the circularity/
self-preference failure this paper warns about (an LLM favoring rewrites produced by its own lineage).
It also disciplines how we *report* the audit — we should not treat a high aggregate agreement as
self-vindicating, and should foreground the per-item behavior. Citing both Upadhyay (rankings largely
preserved) and Clarke–Dietz (but beware circularity) gives the balanced, defensible framing our audit
needs.

## Cite for
- LLM-judge circularity/self-preference risk — the direct justification for our cross-lineage judge.
- High aggregate τ can mask failure modes; report per-item behavior, don't over-trust the correlation.
