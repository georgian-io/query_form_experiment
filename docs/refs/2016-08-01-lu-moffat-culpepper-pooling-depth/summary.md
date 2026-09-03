# The Effect of Pooling and Evaluation Depth on IR Metrics

|  |  |
|---|---|
| **Authors** | Xiaolu Lu, Alistair Moffat, J. Shane Culpepper (RMIT / Melbourne) |
| **Venue / date** | Information Retrieval Journal 19(4), 2016 |
| **Links** | [Springer DOI](https://doi.org/10.1007/s10791-016-9282-6) · [Springer PDF](https://link.springer.com/content/pdf/10.1007/s10791-016-9282-6.pdf) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

Analyzes how **pool depth** (how deep runs are judged) and **evaluation depth** (the k in metrics
like NDCG@k, RBO, RBP) interact to determine whether a metric's scores — and the system comparisons
built on them — are reliable. Distinguishes recall-based metrics (need deep, near-complete judgments)
from utility/top-weighted metrics (tolerate shallow pools). Gives practical guidance on the
depth/reliability trade-off for fixed-depth evaluation (k = 20, 100).

## Relevance to this study

The **graded-judgment complement to [Buckley &
Voorhees](../2004-07-25-buckley-voorhees-incomplete-info/summary.md)** — together they bracket the
Capstone-A metric-blindness argument. Buckley–Voorhees covers the *incomplete/sparse* extreme (our
1-gold ChatDoctor); Lu–Moffat–Culpepper covers how *judgment/evaluation depth* governs what a metric
can resolve, which is the right lens for our **dense** datasets (CUREv1 ~40 golds, TREC-COVID ~494
graded golds). It explains, in general terms, *why* a densely-judged collection can register the
top-10 damage that a shallow/sparse one cannot — the mechanism our sparse-vs-dense contrast
demonstrates.

Also relevant to our metric choice: it treats RBO/RBP (top-weighted) vs recall-based measures
explicitly, backing our decision to report a top-weighted leaderboard metric.

## Cite for
- Judgment/evaluation depth determines metric reliability and what damage a metric can resolve.
- The graded-qrel half of the sparse-vs-dense metric-blindness argument (with Buckley–Voorhees).
