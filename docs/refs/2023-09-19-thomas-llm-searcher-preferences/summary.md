# Large Language Models can Accurately Predict Searcher Preferences

|  |  |
|---|---|
| **Authors** | Paul Thomas, Seth Spielman, Nick Craswell, Bhaskar Mitra (Microsoft) |
| **Venue / date** | arXiv:2309.10621 — v1 2023-09-19 (SIGIR 2024) |
| **Links** | [abs](https://arxiv.org/abs/2309.10621) · [pdf](https://arxiv.org/pdf/2309.10621) · local: `original.pdf` |
| **Type** | Peer-reviewed research |

## Summary

Shows that an LLM, prompted to imitate real searchers' preferences, produces **relevance labels as
accurate as human labellers** and better than third-party crowd workers, at a fraction of the cost —
validated against gold user feedback at Bing on TREC data. Two findings matter beyond the headline:
LLM labels train **better rankers**, and label quality is **sensitive to prompt wording** — "simple
paraphrases" of the prompt change accuracy.

## Relevance to this study

Two distinct hooks:

1. **Validates our LLM-judge methodology (§5.6 intent-drift audit).** We use a cross-lineage LLM to
   judge whether a rewrite changed the query's intent. Thomas et al. is the citation that an LLM can
   stand in for human judgment on relevance/intent well enough to trust — the evidence base our audit
   design leans on.
2. **Independent phrasing-sensitivity result on the judgment side.** Their finding that *paraphrasing
   the prompt* shifts LLM label accuracy is the evaluator-side echo of our whole thesis: meaning-
   preserving wording changes carry real signal. It also flags a caveat for our own audit — the
   drift judge is itself phrasing-sensitive, so we hold its rubric/prompt fixed across conditions.

## Cite for
- LLMs are accurate relevance/intent judges (basis for trusting the drift audit).
- Even meaning-preserving paraphrase changes LLM-judge behavior (phrasing sensitivity; audit caveat).
