# Variations in Relevance Judgments and the Shelf Life of Test Collections

|  |  |
|---|---|
| **Authors** | Andrew Parry, Maik Fröbe, Harrisen Scells, Ferdinand Schlatt, Guglielmo Faggioli, Saber Zerhoudi, Sean MacAvaney, Eugene Yang |
| **Venue / date** | arXiv:2502.20937 — v1 2025-02-28 (SIGIR 2025) |
| **Links** | [abs](https://arxiv.org/abs/2502.20937) · [pdf](https://arxiv.org/pdf/2502.20937) · [html](https://arxiv.org/html/2502.20937v2) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

The modern re-test of [Voorhees 2000](../2000-09-01-voorhees-variations-relevance-judgments/summary.md)
on **neural** retrieval. Re-annotating TREC DL 2019, the authors ask whether the Cranfield principle —
"system rankings are stable even when assessors disagree" — still holds for today's short-document,
graded-scale collections. Findings: **assessor disagreement does not destabilize overall system
rankings** (the principle holds), *but* individual models can show substantial effectiveness swings
under new judgments, and modern, loosely-specified collections may have a limited "shelf life."

## Relevance to this study

The **free-PDF primary source for the leaderboard-stability premise**, and more current than Voorhees
for neural embedders. It lets the writeup assert — with a 2025 citation on neural systems — that
overall rankings do not wander under judgment noise, so our τ reshuffle is attributable to query
form.

It also supplies a **useful nuance we should honor**: while *aggregate* rankings are stable,
*individual models* can be judgment-sensitive. Our reshuffle is precisely a story about individual
models trading places, so this paper both licenses the method (aggregate stability) and cautions
against over-reading any single model's movement (individual sensitivity) — a caveat worth a sentence
in our per-model discussion.

## Cite for
- Contemporary confirmation (neural systems) that system rankings are stable under judgment variation.
- Caveat: individual-model effectiveness can still swing with re-judgment — read per-model moves carefully.
