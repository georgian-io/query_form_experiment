# Related-work reference collection

Sources backing the detailed writeup of the query-form / leaderboard-reshuffle study. Seeded
2026-07-28 from the literature check on the "verbose = how LLMs write queries" claim, then extended,
then expanded with a focused robustness/evaluation pass (4 parallel search agents, deduped).

> **Location note.** This lives at **`docs/refs/`**, not a top-level `refs/`: the sandbox in this
> environment refuses to create new top-level directories (writes silently go to an ephemeral overlay
> and vanish), but subdirectories under the existing `docs/` persist.

## Conventions

- One folder per reference: **`YYYY-MM-DD-<stub>`**, dated by the reference's own **publication
  date**, so the directory sorts chronologically.
- Each folder holds **`original.pdf`** (or `original.html`) and **`summary.md`** (factual summary +
  **relevance note** + a "Cite for" list). ⚠️ marks a non-peer-reviewed source or a reference whose
  original could not be stored (paywalled, no free PDF).
- **33 references currently.**

## Index

| Date | Reference | Type | One-line relevance |
|---|---|---|---|
| 2000-09-01 | [Voorhees — Variations in Relevance Judgments](2000-09-01-voorhees-variations-relevance-judgments/summary.md) | peer ⚠️no-pdf | **Methodology license**: rankings stable under judgment variation → our reshuffle is a query effect. |
| 2004-07-25 | [Buckley & Voorhees — Incomplete Information](2004-07-25-buckley-voorhees-incomplete-info/summary.md) | peer | **Capstone A backbone**: incomplete/sparse judgments distort scores *and* rankings. |
| 2010-11-01 | [Webber, Moffat, Zobel — RBO](2010-11-01-rbo-webber-indefinite-rankings/summary.md) | peer | Methods citation for our top-weighted leaderboard metric. |
| 2016-07-17 | [UQV100 — Query Variability](2016-07-17-uqv100-query-variability/summary.md) | peer ⚠️no-pdf | Human query variability is large; upstream of Penha; kin to our TREC-COVID human fields. |
| 2016-08-01 | [Lu, Moffat, Culpepper — Pooling & Eval Depth](2016-08-01-lu-moffat-culpepper-pooling-depth/summary.md) | peer | Graded-judgment half of the metric-blindness argument. |
| 2021-08-27 | [Zhuang & Zuccon — Typos for BERT Retrieval](2021-08-27-typos-bert-passage-retrieval/summary.md) | peer | Surface/typo endpoint of the query-form spectrum; the contrast class to our semantic rewrites. |
| 2021-11-25 | [Penha et al. — Query Variation Generators](2021-11-25-penha-query-variation-generators/summary.md) | peer | **Closest pre-LLM prior work**: retrieval brittle to meaning-preserving variation (~20% drop). |
| 2022-09-23 | [Promptagator](2022-09-23-promptagator/summary.md) | peer | Q3: synthetic queries resemble real ones only when distribution-matched. |
| 2022-10-13 | [MTEB](2022-10-13-mteb/summary.md) | peer | Leaderboard-construction framework; RTEB's lineage. |
| 2022-12-20 | [HyDE](2022-12-20-hyde/summary.md) | peer | Canonical LLM query→**document** expansion; the "automated rewrite" verbose is *not*. |
| 2023-03-14 | [Query2doc](2023-03-14-query2doc/summary.md) | peer | Pseudo-doc expansion **boosts BM25** — the lexical signature verbose lacks. |
| 2023-04-18 | [Faggioli et al. — Perspectives on LLM Relevance Judgment](2023-04-18-faggioli-perspectives-llm-relevance-judgment/summary.md) | peer | The LLM-as-judge debate + risk taxonomy behind our drift audit. |
| 2023-05-23 | [Ma et al. — Query Rewriting for RAG](2023-05-23-ma-query-rewriting-rag/summary.md) | peer | **Motivating premise**: LLMs rewrite the query in production RAG. |
| 2023-09-19 | [Thomas et al. — LLMs Predict Searcher Preferences](2023-09-19-thomas-llm-searcher-preferences/summary.md) | peer | Validates LLM-judge; paraphrase changes judge accuracy. |
| 2024-01-18 | [Traditional vs LLM Search (Geolocation)](2024-01-18-llm-vs-search-geolocation/summary.md) | peer (n=60) | **Empirical anchor**: LLM querying → longer NL queries (6.06 vs 4.19 terms). |
| 2024-07-09 | [Robust Neural IR — Adversarial & OOD survey](2024-07-09-robust-neural-ir-survey/summary.md) | preprint (survey) | The robustness **taxonomy** (positions us as OOD query variation). |
| 2024-11-13 | [Upadhyay et al. — Large-Scale LLM Relevance Study](2024-11-13-upadhyay-large-scale-llm-relevance/summary.md) | peer | LLM qrels preserve rankings at **τ≈0.89** — a yardstick for our sub-0.9 effects. |
| 2024-11-20 | [DMQR-RAG](2024-11-20-dmqr-rag/summary.md) | preprint | Production rewriting is a **portfolio** of forms, not one verbose restatement. |
| 2024-12-22 | [Clarke & Dietz — LLM Judgment Can't Replace Human](2024-12-22-clarke-dietz-llm-cant-replace-human/summary.md) | preprint | Counterweight: **circularity** risk → justifies our cross-lineage judge. |
| 2025-02-28 | [Shelf Life of Test Collections](2025-02-28-shelf-life-test-collections/summary.md) | peer | Modern (neural) confirmation rankings are stable under judgment variation; per-model caveat. |
| 2025-06-10 | [Granularity Dilemma of Embeddings](2025-06-10-granularity-dilemma-embeddings/summary.md) | peer | Dense/graded qrels expose failures sparse benchmarks miss (metric-blindness support). |
| 2025-06-16 | [Dense Retriever Robustness — Interdisciplinary](2025-06-16-dense-retriever-robustness-interdisciplinary/summary.md) | preprint | Measured robustness is an artifact of **benchmark labeling** — metric-blindness, other angle. |
| 2025-07-09 | [RAG Robustness at the Query Level](2025-07-09-rag-robustness-query-level/summary.md) | peer (wksp) | Peer-reviewed component-level query-form sensitivity (1,092 experiments). |
| 2025-08-11 | [Retrieval Coherence for Semantically Equivalent Queries](2025-08-11-retrieval-coherence-equivalent-queries/summary.md) | preprint | Divergent top-k under paraphrase (≈ our RBO, per-query); a remedy. |
| 2025-08-28 | [Theoretical Limitations of Embedding Retrieval](2025-08-28-theoretical-limitations-embedding-retrieval/summary.md) | preprint | Provable capacity ceiling of single-vector retrieval — theoretical bookend to metric blindness. |
| 2025-09-09 | [Query Expansion in the Age of PLMs/LLMs — Survey](2025-09-09-query-expansion-llm-survey/summary.md) | preprint (survey) | Orientation map; positions HyDE/query2doc/DMQR. |
| 2025-10-08 | [PTEB — Stochastic Paraphrasing Eval](2025-10-08-pteb-stochastic-paraphrasing/summary.md) | preprint | **Closest sibling**: embedding rankings shift under meaning-preserving paraphrase. |
| 2025-10-28 | [RTEB explainer](2025-10-28-rteb-benchmark/summary.md) | ⚠️ blog | The benchmark under study. |
| 2026-02-05 | [SOCi — LLM Queries 6× Longer](2026-02-05-soci-llm-query-length/summary.md) | ⚠️ industry | Magnitude anchor (~23-word LLM query); directional only. |
| 2026-03-02 | [Not All Queries Need Rewriting](2026-03-02-not-all-queries-need-rewriting/summary.md) | preprint | **Mechanistic match**: rewriting corpus-aligned queries hurts via terminology displacement. |
| 2026-04-12 | [How You Ask Matters! — Adaptive RAG Robustness](2026-04-12-how-you-ask-matters-rag-robustness/summary.md) | preprint | Near-identical frame (fix intent, vary form); scale doesn't close the gap. |
| 2026-04-17 | [Robustness of LLM-Based Dense Retrievers](2026-04-17-llm-dense-retriever-robustness/summary.md) | preprint | **Our exact model class**: typo-robust but semantics-fragile (predicts which conditions reshuffle). |
| 2026-05-21 | [Instruction Sensitivity Undermines Embedding Eval](2026-05-21-instruction-sensitivity-embedding-eval/summary.md) | preprint | **Closest analog on the instruction axis**: any model → #1 by prompt choice; justifies our instruction-pinning. |

## Themes (for structuring the writeup)

**A · What "LLM query rewriting" actually is** — HyDE, query2doc, DMQR, QE survey, Ma et al. Mostly
pseudo-document / multi-query / trained-rewriter methods, not first-person elaborated needs.

**B · How long / natural LLM-era queries are** — geolocation (6.06 vs 4.19 terms) + SOCi (⚠️). Sets
the ~20-word norm our verbose doubles.

**C · When rewriting / variation helps vs hurts, and why** — Not-All-Queries (terminology
displacement), Penha (~20% drop), query2doc (BM25 enrichment). The spine of our drift + BM25 controls.

**D · Retrieval/embedding robustness to query form (the sibling cluster)** — the pass's payload:
- *LLM-era, meaning-preserving, near-our-frame*: **How You Ask Matters!**, **Robustness of LLM-Based
  Dense Retrievers** (our model class), **Retrieval Coherence for Equivalent Queries**, **RAG
  Robustness at Query Level**, **PTEB** (siblings §F below).
- *Surface endpoint / contrast*: **Zhuang & Zuccon typos**.
- *Taxonomy*: **Robust Neural IR survey** (OOD vs adversarial).

**E · Evaluation methodology — metrics, judgments, leaderboards** —
- *Rankings stable under judgment noise*: **Voorhees 2000** + **Shelf Life 2025**.
- *Sparse/incomplete & graded judgments → metric blindness*: **Buckley & Voorhees**, **Lu–Moffat–
  Culpepper**, **Granularity Dilemma**, **Dense Retriever Robustness (Interdisciplinary)**.
- *Metric definition*: **RBO**.
- *Benchmark construction / limits*: **MTEB**, **RTEB explainer**, **Theoretical Limitations of
  Embedding Retrieval**.
- *Instruction/prompt sensitivity of embedders*: **Instruction Sensitivity Undermines Embedding
  Eval** (justifies holding our query instruction constant).
- *LLM-as-judge (our drift audit)*: **Faggioli** (debate), **Upadhyay** (rankings preserved τ≈0.89),
  **Clarke & Dietz** (circularity caveat), **Thomas** (LLM judges are phrasing-sensitive too).

**F · Direct siblings (embedding robustness to phrasing)** — **PTEB** (paraphrase → rankings shift),
**Instruction Sensitivity** (prompt → rankings shift), **Retrieval Coherence** (paraphrase → top-k
shift), with **Penha** the pre-LLM antecedent. Position our contribution against these: we add the
*terse/verbose spectrum* and the *sparse-vs-dense metric blind spot*.

## The framings that hold the story together

- **Per-system vs cross-system.** Penha / Not-All-Queries / the robustness cluster measure one
  system's effectiveness drop; our contribution is the *cross-system leaderboard reshuffle* (τ/RBO).
  **PTEB** and **Instruction Sensitivity** are the two prior works that also look at *rankings* under
  phrasing — we extend them with the terse/verbose spectrum and the metric blind spot.
- **Can the metric see it?** Voorhees/Shelf-Life: rankings are stable under judgment *noise*.
  Buckley–Voorhees / Lu–Moffat–Culpepper / Granularity-Dilemma / Interdisciplinary-robustness:
  sparse/coarse judgments *can't resolve* certain differences. Our sparse-vs-dense medical pair
  (ChatDoctor 1-gold vs CUREv1 ~40) is where those two facts collide.

## Surveyed but not added (deduped in the robustness pass)

Considered and deliberately excluded — recorded so the survey is reproducible. Easy to promote in if a
section needs them.

- **Typo-robustness mitigations** (redundant with Zhuang & Zuccon 2108.12139): CharacterBERT +
  Self-Teaching (2204.00716), CAPOT (2304.03401), Multi-Positive Contrastive typos (2403.10939), Dual
  Encoders vs Misspellings (2205.02303).
- **Robustness survey duplicate**: "Robust Information Retrieval" SIGIR'24 tutorial (2406.08891) —
  folded into the 2407.06992 survey summary (same group).
- **Generative-IR OOD**: On the Robustness of Generative IR — OOD (2412.18768) + precursor
  (2306.12756). Different model class (docid-generating), tangential to dense embedders.
- **Instruction-following benchmarks** (covered by Instruction-Sensitivity 2605.22544): FollowIR
  (2403.15246, EMNLP'24), InfoSearch / Beyond Content Relevance (2410.23841, ICLR'25), INSTRUCTIR
  (2402.14334). Promote one if we expand the instruction-axis discussion.
- **Classical query-variation-as-evidence**: Boosting Search Performance Using Query Variations
  (1811.06147, TOIS'19) — overlaps UQV100/Benham theme.
- **Benchmark reliability, adjacent**: Maintaining MTEB (2506.21182), How Hard is it to Rig a
  Benchmark? social-choice (2605.23628, LLM suites not embeddings).
- **LLM-judge extras**: UMBRELA tool paper (2406.06519), Benchmarking LLM Judgment Methods
  (2504.12558), Limitations of Automatic Relevance Assessments (2411.13212).
- **Other**: WebDRO group-reweighting robustness (2310.16605), QPP for query-variant selection in RAG
  (2604.22661).

## Gaps to fill next

- **Conversational query rewriting / decontextualization** — QReCC or TREC CAsT self-contained
  rewriting ("rewriting ≠ lengthening").
- **A specific "verbose queries hurt via extraneous terms" citation** — FIRE 2024 query-token-
  importance paper (ACM-paywalled; no free PDF located).
- **Primary RTEB spec** — replace the blog explainer if an authoritative RTEB paper appears.
- **Missing originals to backfill** — Voorhees 2000 and UQV100 are summary-only (paywalled).
