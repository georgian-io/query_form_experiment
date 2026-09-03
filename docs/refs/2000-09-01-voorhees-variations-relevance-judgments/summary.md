# Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness

|  |  |
|---|---|
| **Authors** | Ellen M. Voorhees (NIST) |
| **Venue / date** | Information Processing & Management 36(5), 2000 (extends SIGIR 1998) |
| **Links** | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306457300000108) · [SIGIR'98 ACM](https://dl.acm.org/doi/10.1145/290941.291017) · [NIST record](https://www.nist.gov/publications/variations-relevance-judgments-and-measurement-retrieval-effectiveness) |
| **Type** | Peer-reviewed research (foundational) · ⚠️ **no free PDF located — original not stored** |

## Summary

The classic result on evaluation stability. Voorhees had TREC topics re-judged by different
assessors and measured how much **system rankings** changed across judgment sets. Despite assessors
disagreeing substantially on individual documents (overlap often ~0.3–0.5), the **relative ranking
of systems was highly stable** — very high Kendall-τ correlations between rankings produced from
different judges. Conclusion: comparative IR evaluation is robust to assessor disagreement, which is
what makes reusable test collections viable.

## Relevance to this study

This is the **methodological license for our entire approach.** We compare *leaderboards* with
Kendall's τ and treat a drop below 0.9 as a real reshuffle. Voorhees establishes the null we rely
on: with the judgments held fixed, system rankings do **not** wander — so when *we* see τ fall to
0.80 or 0.52, it cannot be dismissed as judgment noise; it is caused by the one thing we changed, the
**query phrasing**. Voorhees fixes the queries and varies the judges (rankings stable); we fix the
judges and vary the queries (rankings move). Stating that symmetry up front is the cleanest way to
pre-empt the "maybe it's just noise" objection.

Its modern re-test on neural systems is
[2025-02-28 Shelf Life](../2025-02-28-shelf-life-test-collections/summary.md), which has a free PDF
and reaches the same conclusion (rankings stable) while adding a caveat about individual-model
sensitivity.

## Cite for
- System/leaderboard rankings are stable under relevance-judgment variation (the τ methodology precedent).
- Justification that our observed reshuffle is a query-form effect, not judgment noise.
