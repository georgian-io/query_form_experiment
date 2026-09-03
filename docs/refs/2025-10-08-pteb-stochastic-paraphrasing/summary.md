# PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs

|  |  |
|---|---|
| **Authors** | Manuel Frank, Haithem Afli |
| **Venue / date** | arXiv:2510.06730 — v1 2025-10-08 |
| **Links** | [abs](https://arxiv.org/abs/2510.06730) · [pdf](https://arxiv.org/pdf/2510.06730) · [html](https://arxiv.org/html/2510.06730v2) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Argues that static suites like MTEB let repeated tuning inflate scores and hide real robustness, and
proposes **PTEB**: at evaluation time, stochastically generate **meaning-preserving paraphrases** of
the inputs (LLM-based, human-validated) and aggregate over multiple runs. Across 20 datasets and 25
languages, sentence-encoder performance is **sensitive to token-space changes even when semantics are
fixed**, and rankings shift under paraphrasing; smaller models are not systematically less robust.

## Relevance to this study

The **closest published sibling to our work** — independent, near-simultaneous, and convergent. PTEB
demonstrates on standard embedding benchmarks exactly the phenomenon our **paraphrase** condition
isolates: *meaning-preserving rewording moves embedding results and rankings.* Strong external
corroboration of our Q1 finding that phrasing is a real axis of embedding quality.

Positioning (how ours differs / complements — worth a paragraph in related work):
- **Scope of rewrite.** PTEB studies paraphrase (meaning-preserving). We study a *spectrum* —
  paraphrase **and** terse compression **and** verbose elaboration — and find the effect grows with
  the move, verbose being the most disruptive.
- **The metric-blindness axis is ours alone.** PTEB does not isolate relevance-label density; our
  central result — that a **sparse (1-gold) benchmark cannot see** the elaboration effect a dense one
  registers — is orthogonal to and extends PTEB's robustness message.
- **Goal.** PTEB proposes a debiased *evaluation protocol* (perturb to get robust scores); we run a
  *measurement study* of how rewrite style reshuffles the leaderboard and whether the metric can see
  it. Cite PTEB as convergent evidence and as motivation for perturbed/dynamic embedding evaluation.

## Cite for
- Independent evidence that embedding scores **and rankings** change under meaning-preserving paraphrase.
- Motivation for dynamic/perturbed embedding evaluation (vs static leaderboards).
- The paper to distinguish ourselves from: we add the terse/verbose spectrum and the sparse-vs-dense metric blind spot.
