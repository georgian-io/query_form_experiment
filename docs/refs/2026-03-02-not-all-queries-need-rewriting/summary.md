# Not All Queries Need Rewriting: When Prompt-Only LLM Refinement Helps and Hurts Dense Retrieval

|  |  |
|---|---|
| **Authors** | Varun Kotte |
| **Venue / date** | arXiv:2603.13301 — v1 2026-03-02 |
| **Links** | [abs](https://arxiv.org/abs/2603.13301) · [pdf](https://arxiv.org/pdf/2603.13301) · [html](https://arxiv.org/html/2603.13301) · local: `original.pdf` |
| **Type** | Preprint |

## Summary

Studies prompt-only, single-step LLM query rewriting (rewrite from the query alone, no retrieval
feedback) — the common production pattern — and its effect on **dense** retrieval across three
benchmarks:

- **Strongly domain-dependent:** rewriting **degrades FiQA, improves TREC-COVID, neutral on
  SciFact.** No blanket win.
- **Lexical substitution in 95% of rewrites**; what matters is the *direction*, not whether it
  happens. Harm comes when rewrites **displace domain-specific terminology from queries already
  well-matched** to the corpus (reduced lexical alignment).
- **Selective gating** reduces worst-case regressions but does not reliably beat *never rewriting*.
- Recommends **domain-adaptive post-training** over prompt-only refinement when supervision exists.

## Relevance to this study

The **strongest mechanistic corroboration** we have. Their core result — rewriting a corpus-aligned
query tends to *hurt*, via displaced terminology and reduced lexical alignment — is the same
phenomenon our verbose condition exhibits at leaderboard scale: elaboration adds off-intent scope and
reshuffles who wins.

Two caveats to flag when citing (opportunities, not conflicts):

1. **Different outcome variable.** They measure *absolute NDCG* of a single rewritten query; we
   measure *leaderboard reshuffle (τ/RBO) across many models*. A rewrite can raise average NDCG
   while still reordering which model is best — our axis, not theirs.
2. **TREC-COVID sign.** They report rewriting *helps* TREC-COVID absolute NDCG, whereas TREC-COVID
   is where our synthetic transforms reshuffle the board *most*. Not a contradiction (different
   metric, different rewrite prompts, dense qrels), but exactly the nuance the detailed writeup
   should meet head-on — same dataset, two different questions.

Also supports the BM25 story: "95% lexical substitution, effect depends on direction" is the
fine-grained version of our coarse BM25-shift row.

## Cite for
- Rewriting already-aligned queries hurts via terminology displacement / reduced lexical alignment.
- Prompt-only rewriting is not a blanket win; effects are domain-dependent.
- Contrast of outcome variables (absolute NDCG vs leaderboard reshuffle) — a framing point.
