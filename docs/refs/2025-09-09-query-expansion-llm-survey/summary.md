# Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey

|  |  |
|---|---|
| **Authors** | Minghan Li, Xinxuan Lv, Junjie Zou, Tongna Chen, Chao Zhang, Suchao An, Ercong Nie, Guodong Zhou |
| **Venue / date** | arXiv:2509.07794 — v1 2025-09-09 |
| **Links** | [abs](https://arxiv.org/abs/2509.07794) · [pdf](https://arxiv.org/pdf/2509.07794) · local: `original.pdf` |
| **Type** | Preprint (survey) |

## Summary

A comprehensive survey of **query expansion (QE)** in the PLM/LLM era. Frames QE as the classic
remedy for **vocabulary mismatch** between short, ambiguous queries and diverse corpora, and
organizes modern techniques along four design dimensions: (1) model capabilities unlocked by
PLMs/LLMs (contextualization, controllable generation, instruction-following); (2) where expansion
sits in the pipeline and how it grounds against corpus evidence; (3) learning/alignment strategies;
(4) knowledge incorporation and deployment trade-offs (effectiveness vs cost vs controllability).

## Relevance to this study

The **orientation / positioning reference** — the one-stop map for the related-work section. It
situates HyDE, query2doc, and DMQR (all in this collection) within a single taxonomy, so the writeup
can place our query *transformation* work against the broader query *expansion* literature and be
precise about the difference:

- QE typically **adds material to improve** retrieval; our conditions **restate** the query and we
  ask whether that *reshuffles a leaderboard* — a robustness/measurement question, not an
  effectiveness-improvement one.
- The survey's "vocabulary mismatch" framing is the lexical-axis counterpart to our BM25 control.

Use it to justify scope boundaries (what we are and are not doing) rather than for a specific result.

## Cite for
- A structured map of PLM/LLM query expansion (positioning HyDE / query2doc / DMQR).
- The vocabulary-mismatch framing and the effectiveness-vs-cost-vs-controllability trade-off.
