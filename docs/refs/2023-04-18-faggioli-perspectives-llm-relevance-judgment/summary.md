# Perspectives on Large Language Models for Relevance Judgment

|  |  |
|---|---|
| **Authors** | Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke, et al. (11 authors) |
| **Venue / date** | arXiv:2304.09161 — v1 2023-04-18 (ICTIR 2023) |
| **Links** | [abs](https://arxiv.org/abs/2304.09161) · [pdf](https://arxiv.org/pdf/2304.09161) · local: `original.pdf` |
| **Type** | Peer-reviewed (ICTIR 2023) |

## Summary

The canonical framing paper on whether relevance judging can be delegated to LLMs. Lays out a
human–machine collaboration spectrum for judgment strategies and closes with explicit **"for,"
"against," and "compromise"** positions from the IR community, naming the core risks: **circularity,
bias, and over-leniency.**

## Relevance to this study

The reference point for the debate our **§5.6 intent-drift audit** steps into. When we use an LLM to
judge whether a rewrite changed the query's intent, we are doing LLM-as-judge, and this paper defines
the risk categories our design must answer to. It directly motivates two of our safeguards: judging
with a **cross-lineage** model (to limit circularity/self-preference) and **pinning the judge's rubric
and prompt version** (to limit prompt-driven variance). Cite it to show the audit is designed against
a known risk taxonomy, not naively trusting an LLM.

## Cite for
- The LLM-as-relevance-judge debate and its risk taxonomy (circularity, bias, over-leniency).
- Motivation for our cross-lineage judge and pinned-rubric audit design.
