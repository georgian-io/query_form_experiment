# query_form_experiment

The code and the evidence behind *Your Queries Can Reorder the Leaderboard*.

I explored what happens to a retrieval leaderboard when you change only the *form* of the
queries, holding the corpus and the relevance labels fixed. This repository is the harness that
ran the experiment and the per-query score files it produced. If you have just read the post and
want to see where a number came from, the table below is the index. Every number recomputes from
a clone, offline, with nothing paid:

```bash
uv sync
uv run python scripts/05_analyze.py        # the reproduction gate, model by model
uv run python tools/transform_analysis.py  # reshuffle tables, controls, levelling
```

## Where the post's numbers come from

| In the post | Here | Produced by |
|---|---|---|
| The four query forms (CUREv1 example) | qid `000001` in `results/queries__curev1_en__{condition}__gemini-2.5-pro.json` | `scripts/02_run_transform.py` |
| Reproducing published scores | [Reproduction gate](#the-reproduction-gate) | `scripts/05_analyze.py` |
| The τ table, three rewrites × three datasets | [Reshuffle](#leaderboard-reshuffle), averaged over the four generators | `tools/transform_analysis.py` |
| Figure 1, the board reordered by elaboration | [Rank moves](#the-board-reordered-by-elaboration) | `tools/chart_data.py` → `tools/make_post_figures.py` |
| Figure 2, the per-model ΔNDCG heatmap | `deltas.trec_covid` in the chart data | same |
| The human-phrasing τ table | [Human phrasings](#human-phrasings-the-control-on-llm-style) | `tools/transform_analysis.py`, `tools/chart_data.py` |
| Figure 3, three human phrasings ranked | `human_fields` in the chart data | same |
| Compression improves 19 models; the +0.44 and -0.78 strength correlations | [Levelling](#levelling) | `tools/transform_analysis.py` |
| Figure 4 and the one-gold vs pooled-key table | [The sparse-key arithmetic](#the-sparse-key-arithmetic) | `tools/make_sparse_key_figure.py` |
| The τ ranges by label density | [Reshuffle](#leaderboard-reshuffle), the per-generator tables | `tools/transform_analysis.py` |

Figures live in `docs/figures/`. Regenerate them with:

```bash
uv run python tools/chart_data.py chart_data.json
uv run python tools/make_post_figures.py chart_data.json docs/figures
```

## What was run

Three datasets: ChatDoctor marks one relevant document per query, CUREv1 about forty, 
and TREC-COVID several hundred, graded 0/1/2. Each dataset's queries were rewritten three 
ways (paraphrase, terse, verbose) by four LLMs from four families. The rewriters worked 
independently, so if only one of them produces an effect it shows up as such.

Twenty-one embedding models. All 21 ran on CUREv1 and TREC-COVID. 17 of them ran on ChatDoctor, which
is why its board is shorter. Retrieval is turbopuffer, dense by default, with BM25 and hybrid RRF
available as controls.

The harness also carries AILAStatutes and HumanEval as pilots. They were just used to explore
before moving on to the larger corpora, and their runs are still in `results/`, but the post does
not use them.

## The reproduction gate

The first task was to tune the harness so that it could reproduce each model's published NDCG@10 on 
the *original* queries from each same dataset.

| dataset | published results | within 0.01 | worst absolute delta |
|---|---|---|---|
| ChatDoctor | 16 | 16 | 0.00493 (`voyage-4-large`) |
| CUREv1 | 17 | 16 | 0.03345 (`text-embedding-004`) |
| TREC-COVID | 8 | 5 | 0.10769 (`bge-base-en-v1.5`) |
| AILAStatutes (pilot) | 23 | 22 | 0.01274 (`bge-m3`) |
| HumanEval (pilot) | 22 | 22 | 0.00328 (`embed-multilingual-v3.0`) |

81 of 86 pairs match within 0.01. The three TREC-COVID misses are all `bge-*-en-v1.5` prefix models and
they all miss low by a similar amount, so the *ordering* survives - my TREC-COVID board
reproduces the published one at τ = 1.000, an exact order match. The five prefix-free models on
that dataset land within 0.003. I think the bge gap is a prompting difference in how the published
run handled those models, but I have not closed it and I don't want to explain it away, so I'll keep it 
on the books as a miss.

## Leaderboard reshuffle

Kendall's τ between each transformed-query leaderboard and the same dataset's original human-query
leaderboard, dense retrieval, averaged over the four generators. This is the post's headline table:

| rewrite | ChatDoctor (1 gold) | CUREv1 (~40) | TREC-COVID (graded) |
|---|---|---|---|
| paraphrase | 0.915 | 0.957 | 0.900 |
| terse | 0.905 | 0.888 | 0.748 |
| verbose | 0.938 | 0.834 | 0.607 |

Averaging hides whether the generators agree, so the per-generator numbers follow, with RBO
(p = 0.9) alongside. Board size differs by dataset because not every model has a published anchor
on every one.

ChatDoctor, 17-model board, 5,591 queries:

| condition | generator | τ | RBO |
|---|---|---|---|
| paraphrase | gemini-2.5-pro | +0.9265 | 0.9101 |
| paraphrase | claude-opus-4-8 | +0.9265 | 0.9371 |
| paraphrase | gpt-5.1 | +0.8971 | 0.9101 |
| paraphrase | llama-3.3-70b | +0.9118 | 0.9101 |
| terse | gemini-2.5-pro | +0.9118 | 0.9101 |
| terse | claude-opus-4-8 | +0.8971 | 0.9011 |
| terse | gpt-5.1 | +0.9118 | 0.9002 |
| terse | llama-3.3-70b | +0.8971 | 0.8980 |
| verbose | gemini-2.5-pro | +0.9706 | 0.9952 |
| verbose | claude-opus-4-8 | +0.9412 | 0.9371 |
| verbose | gpt-5.1 | +0.9118 | 0.9088 |
| verbose | llama-3.3-70b | +0.9265 | 0.9713 |

CUREv1, 21-model board, 2,000 queries:

| condition | generator | τ | RBO |
|---|---|---|---|
| paraphrase | gemini-2.5-pro | +0.9524 | 0.9947 |
| paraphrase | claude-opus-4-8 | +0.9619 | 0.9933 |
| paraphrase | gpt-5.1 | +0.9524 | 0.9894 |
| paraphrase | llama-3.3-70b | +0.9619 | 0.9933 |
| terse | gemini-2.5-pro | +0.8857 | 0.9604 |
| terse | claude-opus-4-8 | +0.8857 | 0.9604 |
| terse | gpt-5.1 | +0.8952 | 0.9762 |
| terse | llama-3.3-70b | +0.8857 | 0.9604 |
| verbose | gemini-2.5-pro | +0.8476 | 0.9135 |
| verbose | claude-opus-4-8 | +0.8571 | 0.9153 |
| verbose | gpt-5.1 | +0.8286 | 0.8783 |
| verbose | llama-3.3-70b | +0.8000 | 0.8762 |

TREC-COVID, 21-model board, 50 topics. With only 50 queries, τ carries a seeded 1,000-resample
query-level bootstrap CI, and I only count a τ below 0.9 if its interval excludes 0.9:

| condition | generator | τ | 95% CI | RBO |
|---|---|---|---|---|
| paraphrase | gemini-2.5-pro | +0.9048 | [+0.781, +0.924] | 0.9244 |
| paraphrase | claude-opus-4-8 | +0.9048 | [+0.762, +0.914] | 0.9041 |
| paraphrase | gpt-5.1 | +0.9333 | [+0.771, +0.924] | 0.9098 |
| paraphrase | llama-3.3-70b | +0.8571 | [+0.762, +0.905] | 0.8817 |
| terse | gemini-2.5-pro | +0.7238 | [+0.619, +0.781] | 0.8222 |
| terse | claude-opus-4-8 | +0.7429 | [+0.648, +0.810] | 0.7808 |
| terse | gpt-5.1 | +0.8095 | [+0.695, +0.848] | 0.8983 |
| terse | llama-3.3-70b | +0.7143 | [+0.609, +0.752] | 0.7827 |
| verbose | gemini-2.5-pro | +0.6000 | [+0.457, +0.667] | 0.5971 |
| verbose | claude-opus-4-8 | +0.6952 | [+0.524, +0.724] | 0.6620 |
| verbose | gpt-5.1 | +0.5238 | [+0.314, +0.619] | 0.6013 |
| verbose | llama-3.3-70b | +0.6095 | [+0.486, +0.657] | 0.5789 |

I see a pattern across the three tables. Verbose is the gentlest condition
on ChatDoctor and the harshest on the two densely-labelled datasets, and all four generators agree
on that ordering, so I don't read it as one rewriter's quirk. This isn't quite a
controlled ablation though since the datasets differ in more than their label
density.

### The board reordered by elaboration

TREC-COVID, 21 models, ranked by the benchmark's own questions and then by the generator-averaged
verbose rewrite of those same questions. This is Figure 1:

| question | verbose | model |
|---|---|---|
| 1 | 3 | voyage-3-large |
| 2 | 6 | embed-multilingual-v3.0 |
| 3 | 5 | voyage-4-large |
| 4 | 2 | text-embedding-004 |
| 5 | 7 | voyage-4 |
| 6 | 8 | voyage-3.5-int8-512 |
| 7 | 4 | text-embedding-3-large |
| 8 | 10 | gemini-embedding-001 |
| 9 | 9 | voyage-4-lite |
| 10 | 13 | text-embedding-3-small |
| 11 | 15 | embed-v4.0 |
| 12 | 12 | bge-base-en-v1.5 |
| 13 | 16 | bge-small-en-v1.5 |
| 14 | 14 | bge-large-en-v1.5 |
| 15 | 1 | qwen3-embedding-8b |
| 16 | 11 | voyage-law-2 |
| 17 | 18 | multi-qa-MiniLM-L6-cos-v1 |
| 18 | 20 | bge-m3 |
| 19 | 17 | all-mpnet-base-v2 |
| 20 | 19 | all-MiniLM-L12-v2 |
| 21 | 21 | all-MiniLM-L6-v2 |

Three models gain three or more places and four lose three or more. `qwen3-embedding-8b` goes 15th
to 1st. Five of the original top six move down.

## Human phrasings: the control on LLM style

Every rewrite above was written by an LLM, so the obvious objection is that the result is about
how language models write. Since each TREC-COVID topic also ships both of a human-written
keyword `query` and a paragraph-length `narrative` alongside the `question` the benchmark scores, 
I can use these as a check without involving LLMs.

τ against the `question` board, over the 20 models that have all three human phrasings (`bge-m3`
has no human-field run):

| query form | τ vs `question` |
|---|---|
| `keyword` (human) | +0.8211 |
| terse (LLM) | +0.7500 |
| `narrative` (human) | +0.7263 |
| verbose (LLM) | +0.6526 |

A human-written paragraph reorders the board at τ = 0.726 with no change in meaning and no LLM
involved. The LLM rewrites are each somewhat harsher than the human form they imitate, so my read
is that they overstate the effect a little without creating it. The two extremes, `keyword`
against `narrative`, sit at +0.7158, no further apart than either is from the middle form, which
makes me doubt that length alone explains this.

The LLM τ values above are the mean of the four per-generator τ values on that 20-model board,
which is why they differ slightly from the 21-model figures in the reshuffle tables.

Comparing boards is one thing. Matching the shifts model by model is stricter:

| human shift | synthetic shift | Spearman ρ | p | sign agreement |
|---|---|---|---|---|
| `question` → `query` (keyword) | terse | +0.749 | <0.001 | 80% |
| `question` → `narrative` | verbose | +0.671 | 0.001 | 45% |

Keyword compression is well mimicked. Elaboration is only partly mimicked - sign agreement near
chance means the LLM elaboration and a human's both reshuffle hard but land the per-model effect
differently. That is a caveat on the verbose numbers specifically and it does not touch the terse
ones.

## Levelling

Elaboration costs the strongest models the most and lifts several of the weakest. Going the other
way, compressing `narrative` back to `question` improves 19 of the 20 models with human phrasings,
by an average of +0.1098 NDCG@10.

Correlating a change score against the same baseline it was computed from invites regression to
the mean, so I also measured strength on a *separate* dataset and report both readings:

| shift | strength measured on | Spearman ρ | p | n |
|---|---|---|---|---|
| TREC-COVID verbose | CUREv1 (independent) | -0.779 | <0.0001 | 21 |
| TREC-COVID verbose | its own baseline (naive) | -0.887 | <0.0001 | 21 |
| CUREv1 verbose | TREC-COVID (independent) | -0.310 | 0.17 | 21 |
| CUREv1 verbose | its own baseline (naive) | -0.278 | 0.22 | 21 |
| TREC-COVID human narrative | CUREv1 (independent) | -0.444 | 0.05 | 20 |
| TREC-COVID human narrative | its own baseline (naive) | -0.565 | 0.009 | 20 |
| TREC-COVID compression, narrative → question | CUREv1 (independent) | +0.444 | 0.05 | 20 |

From what I can see, some of the effect is that artifact and most of it isn't, at least on TREC-COVID,
where the independent measure holds at -0.779. On CUREv1 it doesn't reach significance either way.
So I wouldn't state this as a general law - it is one dataset's result, with a second dataset
declining to confirm it.


## Controls not in the post

Two further checks, both out of `tools/transform_analysis.py`, on whether the verbose result is a
property of the queries or an artifact of something else.

The first is BM25, which isolates surface-word change since a lexical scorer only sees tokens.
ΔNDCG@10 against the human-query BM25 run, averaged over the four generators:

| dataset | baseline | paraphrase | terse | verbose |
|---|---|---|---|---|
| ChatDoctor | 0.3277 | -0.021 | +0.054 | -0.021 |
| CUREv1 | 0.3555 | -0.062 | +0.009 | -0.005 |
| TREC-COVID | 0.6083 | -0.057 | +0.097 | +0.026 |

Verbose leaves BM25 near flat on all three while moving the dense board hard. I read that as
semantic broadening - the tokens barely change, but the meaning the dense models see does. Terse
on TREC-COVID is the opposite case. BM25 climbs 0.097 there because the rewrite really is more
keyword-like.

The second is a cross-lineage LLM audit, with the judge never drawn from the rewriter's own
family, flagging rewrites that changed the underlying information need. Pooled drift rate against
a 15% threshold, n = 120 sampled queries per cell (50 on TREC-COVID):

| dataset | paraphrase | terse | verbose |
|---|---|---|---|
| ChatDoctor | 1.9% | 17.7% | 26.3% |
| CUREv1 | 3.1% | 10.0% | 32.4% |
| TREC-COVID | 5.0% | 8.0% | 27.9% |

Verbose breaches the threshold, so drift is a live worry, but it is generator-split. gpt-5.1 and
llama drift hard on verbose (52.5% and 56.7% on CUREv1) while claude (6.0-7.5%) and gemini
(12.6-14.9%) elaborate faithfully. The faithful lineages still move the dense board about as far
as the drifting ones - claude's verbose gives τ 0.857 on CUREv1 and 0.695 on TREC-COVID, against
0.962 and 0.905 for its own paraphrase. So far as I can tell the effect survives the drift
control.

Per-generator rates are in `results/audit__{dataset}__{condition}.json`.

## Data

`results/` contains 1,386 files - every number from the experiment recomputes without
re-running any pipelines. The `dense__*` and `bm25__*`
files carry run metadata and a per-query map of qid to NDCG@10. The `queries__*` files carry the
generated rewrites themselves, and the `audit__*` files carry the drift judgments.

## Running it

```bash
uv sync
uv run pytest        # unit tests; integration tests are deselected by default

uv run python scripts/01_build_index.py   --dataset aila_statutes --model gemini-embedding-001
uv run python scripts/03_run_retrieval.py --dataset aila_statutes --model gemini-embedding-001 --mode dense
```

`--mode` also takes `bm25` and `hybrid`; `--condition` and `--generator` select a rewrite.
`scripts/05_analyze.py` and `tools/transform_analysis.py` take dataset names as positional
arguments and default to `chatdoctor curev1_en trec_covid`.
