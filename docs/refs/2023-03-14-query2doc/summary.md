# Query2doc: Query Expansion with Large Language Models

|  |  |
|---|---|
| **Authors** | Liang Wang, Nan Yang, Furu Wei (Microsoft) |
| **Venue / date** | arXiv:2303.07678 — v1 2023-03-14 (EMNLP 2023) |
| **Links** | [abs](https://arxiv.org/abs/2303.07678) · [pdf](https://arxiv.org/pdf/2303.07678) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

query2doc few-shot-prompts an LLM to generate a **pseudo-document** for a query, then expands the
query by concatenating it with that pseudo-document. Reported gains: **BM25 +3% to +15%** on
MS-MARCO / TREC-DL with no fine-tuning, plus improvements to strong dense retrievers in- and
out-of-domain. The pseudo-documents "often contain highly relevant information that can aid in query
disambiguation."

## Relevance to this study

The second canonical automated LLM query-expansion method — and again the output is a
**pseudo-document appended to the query**, not a rephrased natural-language question. Reinforces the
distinction that motivated our verbose-wording fix.

Sharper contrast on mechanism: query2doc **boosts BM25** because it injects corpus-like lexical
terms. **Our verbose condition leaves BM25 flat** — so verbose is *not* query2doc-style lexical
enrichment; it is semantic broadening. This paper is the clean foil for our "BM25 shift" row: the
expected lexical-expansion signature, which verbose conspicuously lacks.

## Cite for
- The "expansion adds lexical/corpus terms → BM25 rises" baseline (our verbose does not do this).
- Automated LLM query expansion = pseudo-document, not elaborated question.
