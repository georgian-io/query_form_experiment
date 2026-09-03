# Evaluating the Robustness of Retrieval Pipelines with Query Variation Generators

|  |  |
|---|---|
| **Authors** | Gustavo Penha, Arthur Câmara, Claudia Hauff (TU Delft) |
| **Venue / date** | arXiv:2111.13057 — v1 2021-11-25 (ECIR 2022) |
| **Links** | [abs](https://arxiv.org/abs/2111.13057) · [pdf](https://arxiv.org/pdf/2111.13057) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

The **closest prior work** to ours. Builds a taxonomy of meaning-preserving query variations from
the UQV100 dataset of real user reformulations, then auto-generates variations in each category and
measures retrieval robustness. Transformation types: four **syntax-changing** (misspelling,
naturality, word-order, **paraphrasing**) and two **semantics-changing** (**generalization /
specialization**, aspect change). Result: retrieval pipelines (incl. BERT re-rankers) are **not
robust** — effectiveness drops **~20% on average** across two datasets and two tasks.

## Relevance to this study

Direct antecedent, and a source of shared vocabulary:

- Their **paraphrasing** type ≈ our **paraphrase** condition; their **generalization/
  specialization** ≈ our **verbose broadening** (and the intent-drift we measure). We can adopt their
  taxonomy terms rather than reinventing them.
- Their UQV100-derived human variations are the same idea as our **TREC-COVID human phrasing fields**
  (keyword / question / narrative): real, meaning-preserving reformulations as a robustness probe.

**Key distinction to state explicitly.** Penha et al. measure the *absolute effectiveness drop of a
single pipeline* under query variation. We measure something they do not: whether variation
**reshuffles the leaderboard across models** (τ/RBO), and whether a sparse vs dense answer key can
even *see* the effect. Their ~20% drop is the per-system view; our τ is the cross-system view. This
is the cleanest "prior work established X, we add Y" hinge in the related-work section.

## Cite for
- Prior evidence that neural retrieval is brittle to semantics-preserving query variation (~20% drop).
- A validated taxonomy of query transformations (paraphrase; generalization/specialization).
- Per-system-effectiveness framing, to contrast with our cross-system-leaderboard framing.
