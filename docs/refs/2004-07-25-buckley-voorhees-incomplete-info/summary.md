# Retrieval Evaluation with Incomplete Information

|  |  |
|---|---|
| **Authors** | Chris Buckley, Ellen M. Voorhees |
| **Venue / date** | SIGIR 2004 (pp. 25–32), Sheffield UK — 2004-07-25 |
| **Links** | [ACM DOI](https://dl.acm.org/doi/10.1145/1008992.1009000) · [NIST PDF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469) · local: `original.pdf` (NIST) |
| **Type** | Peer-reviewed research (foundational, 800+ citations) |

## Summary

The foundational study of what **incomplete relevance judgments** do to IR evaluation. Buckley &
Voorhees show that standard measures (MAP, P@k, NDCG) are **not robust** when the judgment set is
substantially incomplete: as relevant documents go unjudged, system *scores* and — worse — system
*rankings* become unreliable. They introduce **bpref**, a measure designed to stay stable under
incomplete judgments and to correlate with the standard measures when judgments are complete.

## Relevance to this study

The **theoretical backbone for Capstone A (the metric blind spot)** — the piece the reframed brief
was missing (flagged as a gap in the old index). Our ChatDoctor result is an *extreme case* of their
thesis: with **one gold document per query**, the answer key is maximally incomplete, so NDCG@10 can
only see where that single doc lands and is blind to the off-intent documents an elaborated query
pulls into ranks 2–10. Buckley & Voorhees establish, in general, that **sparse/incomplete judgments
distort system comparison** — precisely the mechanism by which a 1-gold benchmark reports "verbose is
harmless" while a densely-judged one shows it is the most disruptive move.

This lets the writeup ground the blind-spot argument in 20 years of IR-evaluation theory rather than
in our data alone. Pair with the pooling-depth literature (see index gaps) for the graded-judgment
angle.

## Cite for
- Standard IR measures are unreliable under incomplete relevance judgments (scores *and* rankings).
- Sparse answer keys distort system comparison — the general form of our sparse-vs-dense finding.
- bpref as the classic robust-to-incompleteness measure.
