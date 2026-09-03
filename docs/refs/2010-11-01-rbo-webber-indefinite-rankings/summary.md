# A Similarity Measure for Indefinite Rankings (Rank-Biased Overlap, RBO)

|  |  |
|---|---|
| **Authors** | William Webber, Alistair Moffat, Justin Zobel (University of Melbourne) |
| **Venue / date** | ACM TOIS 28(4), Article 20 — November 2010 |
| **Links** | [ACM DOI](https://dl.acm.org/doi/10.1145/1852102.1852106) · [author PDF](http://www.williamwebber.com/research/papers/wmz10_tois.pdf) · local: `original.pdf` |
| **Type** | Peer-reviewed research (the canonical RBO reference) |

## Summary

Defines **Rank-Biased Overlap (RBO)**, a similarity measure for comparing two ranked lists that are
**non-conjoint** (different item sets), **incomplete** (only prefixes known), and **top-weighted**
(agreement near the top matters more than at the tail). RBO ∈ [0, 1]; a persistence parameter
**p** controls top-weighting (higher p = deeper before agreement stops mattering). Built on a simple
probabilistic user model, with a base/max bound giving a well-defined value from finite prefixes.

## Relevance to this study

The **methods citation for one of our two headline metrics.** We report leaderboard agreement with
both **Kendall's τ** (rank correlation, all positions equal) and **RBO** (top-weighted). RBO is the
right tool when we care most about *who is near the top* of the leaderboard rather than exact
ordering deep in the tail — the practically relevant question when picking a retriever.

Two points for the methods section:
- **τ and RBO answer different questions**; reporting both is deliberate. τ can move while the top
  stays put, and vice versa. RBO(p) makes the top-weighting explicit and tunable.
- Cite the **p value** we use (top-weighting choice) with this paper as the definition.

## Cite for
- Formal definition and properties of RBO (top-weighted, non-conjoint, incomplete rankings).
- Justification for reporting a top-weighted leaderboard-similarity measure alongside Kendall τ.
