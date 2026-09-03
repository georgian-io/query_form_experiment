# UQV100: A Test Collection with Query Variability

|  |  |
|---|---|
| **Authors** | Peter Bailey, Alistair Moffat, Falk Scholer, Paul Thomas |
| **Venue / date** | SIGIR 2016 (short paper), Pisa — 2016-07-17 |
| **Links** | [ACM DOI](https://dl.acm.org/doi/10.1145/2911451.2914671) · [dataset (figshare)](https://melbourne.figshare.com/articles/dataset/UQV100_An_IR_Test_Collection_With_Query_Variability/3180694) · [Moffat abstract](https://people.eng.unimelb.edu.au/ammoffat/abstracts/bmst16sigir.html) |
| **Type** | Peer-reviewed resource paper · ⚠️ **no free PDF located — original not stored** (dataset is open on figshare) |

## Summary

Builds an IR test collection that deliberately captures **query variability**. For 100 information
needs (TREC Web Track backstories), crowd workers each wrote the query they *would* issue, yielding
**10,835 queries → 5,764 unique variations** after normalization/spell-correction, plus per-worker
effort estimates. The point: a single information need produces a wide spread of human phrasings, and
a test collection should represent that spread rather than one canonical query.

## Relevance to this study

The empirical foundation for the idea that **one need has many faithful phrasings** — the premise
behind both our transform conditions and, especially, our **TREC-COVID human fields** (keyword /
question / narrative), which are a small, curated instance of exactly the UQV100 phenomenon. UQV100
is also the dataset from which [Penha et al.](../2021-11-25-penha-query-variation-generators/summary.md)
derived their query-variation taxonomy, so it sits one level upstream of our closest prior work.

Framing use: UQV100 shows human phrasing variance is large and real; our contribution asks what that
variance does to a *leaderboard* (not just to a single system's score, which is where the
UQV/robustness line has focused).

## Cite for
- Human query variability for a fixed need is large (thousands of distinct phrasings for 100 needs).
- Precedent/motivation for treating query phrasing as a first-class axis; source of the Penha taxonomy.
