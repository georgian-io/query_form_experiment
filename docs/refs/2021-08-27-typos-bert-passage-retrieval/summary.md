# Dealing with Typos for BERT-based Passage Retrieval and Ranking

|  |  |
|---|---|
| **Authors** | Shengyao Zhuang, Guido Zuccon (University of Queensland) |
| **Venue / date** | arXiv:2108.12139 — v1 2021-08-27 (EMNLP 2021) |
| **Links** | [abs](https://arxiv.org/abs/2108.12139) · [pdf](https://arxiv.org/pdf/2108.12139) · [ACL Anthology](https://aclanthology.org/2021.emnlp-main.225.pdf) · local: `original.pdf` |
| **Type** | Peer-reviewed (EMNLP 2021) |

## Summary

The canonical demonstration that **keyword typos cause large drops** in dense-retrieval and BERT
re-ranker effectiveness, with the root cause traced to **WordPiece tokenization** — a single mistyped
token reshapes the post-tokenization sequence. Proposes typos-aware training to restore robustness.

## Relevance to this study

The **surface-level bound on the query-form-sensitivity spectrum** and the clearest *contrast class*
to our study. Zhuang & Zuccon change the query at the **character** level (corpus and labels fixed)
and show retrieval degrades; we change it at the **semantic** level (paraphrase / terse / verbose) and
show the leaderboard reshuffles. Framing the spectrum — typo → keyword compression → paraphrase →
elaboration — lets us position our conditions against a well-known endpoint and stress that our effect
is *not* a tokenization artifact (our BM25 control already rules out the lexical/surface explanation
for verbose).

## Cite for
- Surface (typo/tokenization) perturbations degrade dense retrieval — the character-level endpoint of the spectrum.
- Contrast anchor: our disruption is semantic, not surface (supported by our flat-BM25 verbose control).
