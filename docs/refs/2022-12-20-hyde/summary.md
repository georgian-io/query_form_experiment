# Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)

|  |  |
|---|---|
| **Authors** | Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan |
| **Venue / date** | arXiv:2212.10496 — v1 2022-12-20 (ACL 2023) |
| **Links** | [abs](https://arxiv.org/abs/2212.10496) · [pdf](https://arxiv.org/pdf/2212.10496) · local: `original.pdf` |
| **Type** | Peer-reviewed research (widely cited) |

## Summary

HyDE is the canonical "let an LLM rewrite the query into a longer text" method. Given a query, it
zero-shot instructs an instruction-following LLM to **generate a hypothetical document** — a
plausible-but-possibly-false answer passage. An unsupervised encoder (Contriever) embeds that
generated document, and the resulting vector retrieves real documents by similarity. The dense
encoder's bottleneck filters out fabricated details. Beats unsupervised Contriever and rivals
fine-tuned retrievers across web search, QA, and fact verification, in several languages.

## Relevance to this study

Reference point for the wording fix to the "verbose" condition. When people say "LLMs rewrite
queries into longer text," **HyDE is usually what they mean — and it produces a hypothetical
*answer document*, not an elaborated first-person *question*.** Our verbose condition is an expanded
*information need*, a different object. HyDE lets the writeup draw that distinction explicitly.

Second hook: HyDE's own framing — the generated document "captures relevance patterns but is unreal
and may contain false details" — is the same failure surface as our **intent-drift** measurement.

## Cite for
- Definition of LLM query→document expansion (contrast with our query→need expansion).
- Precedent that LLM-generated retrieval text is longer and may carry fabricated/off-intent content.
