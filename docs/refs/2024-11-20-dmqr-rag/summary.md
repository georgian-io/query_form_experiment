# DMQR-RAG: Diverse Multi-Query Rewriting for RAG

|  |  |
|---|---|
| **Authors** | Zhicong Li, Jiahao Wang, Zhishu Jiang, Hangyu Mao, Zhongxia Chen, Jiazhen Du, Yuanxing Zhang, Fuzheng Zhang, Di Zhang, Yong Liu (Kuaishou et al.) |
| **Venue / date** | arXiv:2411.13154 — v1 2024-11-20 |
| **Links** | [abs](https://arxiv.org/abs/2411.13154) · [pdf](https://arxiv.org/pdf/2411.13154) · local: `original.pdf` |
| **Type** | Preprint (academic + industry validation) |

## Summary

DMQR-RAG argues a single rewrite is not enough: it defines **four rewriting strategies operating at
different information levels**, so queries carrying different amounts of information retrieve a
*diverse* set of documents. An adaptive selector picks the minimal set of rewrites per query.
Validated in academic and production settings.

## Relevance to this study

Third data point that "how LLMs rewrite queries" is **not one canonical move** — here it is an
explicit *portfolio* of rewrites at varying information levels. Supports softening our verbose
clause: production query rewriting fans out into multiple forms, of which a single verbose
elaboration is only one.

Conceptual bridge: DMQR's "**queries with varying information quantities retrieve diverse
documents**" is, from the model-developer side, the same lever we probe from the evaluation side.
Our terse↔verbose axis *is* an information-quantity axis; DMQR treats that variation as a feature to
exploit, we show it as a benchmark-visibility problem.

## Cite for
- Query rewriting in production is a diverse multi-strategy space, not one verbose restatement.
- The information-quantity-of-query lever, viewed from the retrieval-improvement side.
