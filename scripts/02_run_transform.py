"""Generate transformed query sets (§5.2, Phase 2).

Produces one rewritten query per original, cached on every field that can change the output.
Nothing here touches retrieval or evaluation -- the transform layer is generator-agnostic on one
side and blind to the retrieval engine on the other (§2), so a new generator or condition drops
in without any downstream change.

Generation is sampled (temperature 0.7), so the cache is what makes a run reproducible at all: a
second execution must return the same queries, not merely similar ones.

    uv run python scripts/02_run_transform.py --dataset aila_statutes --condition paraphrase \\
        --generator gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_datasets, load_experiment  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.transform.cache import (  # noqa: E402
    Generation,
    GenerationCache,
    GenerationKey,
    flatten_id,
)
from src.transform.conditions import get_condition  # noqa: E402
from src.transform.generators import get_generator  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", default="paraphrase")
    parser.add_argument("--generator", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N queries, for smoke tests")
    parser.add_argument("--sample", type=int, default=None,
                        help="paraphrase a deterministic seeded subsample of N queries. Valid "
                             "because the transform analysis is paired within our own runs; the "
                             "reproduction gate over the full set is a separate check.")
    parser.add_argument("--shard", default=None,
                        help="'i/n': process only qids whose index %% n == i, and DO NOT write "
                             "the output file (cache-fill only). Run n shards concurrently to "
                             "parallelize a slow generator over disjoint queries, then one plain "
                             "(unsharded) pass materializes the full file from cache.")
    args = parser.parse_args()

    experiment = load_experiment()
    dataset_config = load_datasets().dataset(args.dataset)
    generator_config = next(
        (g for g in experiment.generators if g.id == args.generator), None
    )
    if generator_config is None:
        known = ", ".join(g.id for g in experiment.generators)
        print(f"unknown generator {args.generator!r}; configured: {known}")
        return 1

    dataset = load_dataset(args.dataset, dataset_config)
    dataset.validate()
    queries = dataset.queries()

    if args.sample is not None:
        # Sorted first so selection is independent of dict order, then a seeded draw so every
        # generator paraphrases the SAME subsample and reruns reproduce it exactly.
        rng = random.Random(experiment.seed)
        qids = sorted(rng.sample(sorted(queries), min(args.sample, len(queries))))
    else:
        qids = list(queries)[: args.limit]

    shard_i = shard_n = None
    if args.shard is not None:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        qids = [q for idx, q in enumerate(sorted(qids)) if idx % shard_n == shard_i]

    condition = get_condition(args.condition)
    generator = get_generator(generator_config, experiment.temperature, experiment.seed)
    cache = GenerationCache()

    print(f"{args.dataset}: {len(qids)} queries | {condition.prompt_version} "
          f"(sha {condition.prompt_sha}) | {generator_config.id} "
          f"(lineage {generator_config.lineage}, temp {experiment.temperature})")

    transformed, hits, versions, skipped = {}, 0, set(), {}
    for i, qid in enumerate(qids, start=1):
        key = GenerationKey(
            dataset=args.dataset,
            qid=qid,
            condition=condition.name,
            generator=generator_config.id,
            prompt_version=condition.prompt_version,
            prompt_sha=condition.prompt_sha,
            temperature=experiment.temperature,
            seed=experiment.seed,
            max_output_tokens=generator_config.max_output_tokens,
        )
        cached = cache.get(key)
        if cached is None:
            prompt = condition.build_prompt(queries[qid], dataset_config.domain_hint)
            # A single query the provider refuses (empty completion after retries) or otherwise
            # hard-fails must not kill a 5,000-query run. Fall back to the ORIGINAL query text,
            # so every generator yields the same 5,591 aligned queries -- alignment the §1.2
            # cross-generator comparison depends on. A fallback query is a no-op transform (it
            # equals the human query), so it contributes zero ΔNDCG: the effect is conservatively
            # diluted, never inflated. The fallback is cached (so the count is stable across
            # reruns) and its qid + cause are recorded in the output's `skipped` map.
            try:
                completion = generator.generate(prompt)
                text = condition.postprocess(completion.text)
                model_version = completion.model_version
            except Exception as exc:  # noqa: BLE001 -- boundary: isolate one query's failure
                skipped[qid] = f"{type(exc).__name__}: {exc}"
                print(f"  FALLBACK {qid} (kept original): {type(exc).__name__}: {str(exc)[:100]}")
                text = queries[qid]
                model_version = f"FALLBACK_ORIGINAL::{generator_config.id}"
            cached = Generation(
                text=text, raw=text, model_version=model_version, key=key.__dict__
            )
            cache.put(key, cached)
            if i % 10 == 0 or i == len(qids):
                print(f"  generated {i}/{len(qids)}")
        else:
            hits += 1
        transformed[qid] = cached.text
        versions.add(cached.model_version)

    if skipped:
        print(f"WARNING: {len(skipped)}/{len(qids)} queries fell back to original (generator "
              f"refused/failed); e.g. {list(skipped)[:5]}")

    if shard_n is not None:
        # Cache-fill worker: its transformed dict is only this shard's slice, so it must NOT
        # write the output file. A later unsharded pass reads the whole set from cache and writes.
        print(f"shard {shard_i}/{shard_n} done: {len(transformed)} cached "
              f"({hits} hits, {len(skipped)} fallbacks)")
        return 0

    empty = [q for q, t in transformed.items() if not t.strip()]
    if empty:
        print(f"WARNING: {len(empty)} empty rewrites: {empty[:5]}")

    # A paraphrase should be about the length of its source. A large drift means the condition
    # has quietly become a different transform -- summarization, or a truncated generation --
    # and that would be attributed to phrasing in the ΔNDCG.
    by_qid = {q: len(t) / len(queries[q]) for q, t in transformed.items() if queries[q]}
    ratios = sorted(by_qid.values())
    median = ratios[len(ratios) // 2]
    print(f"length ratio vs original: median {median:.2f}  "
          f"min {ratios[0]:.2f}  max {ratios[-1]:.2f}")

    # Per-query outliers catch a few bad generations that a healthy median would hide.
    collapsed = sorted(q for q, r in by_qid.items() if r < 0.35)
    if collapsed:
        print(f"WARNING: {len(collapsed)} rewrites under 35% of source length: {collapsed[:5]}")
    if condition.name == "paraphrase" and not 0.7 <= median <= 1.4:
        print(f"WARNING: median length ratio {median:.2f} is far from 1.0 -- this is no longer a "
              "paraphrase but a different transform. Check max_output_tokens, and whether the "
              "model is compressing rather than restating.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = (RESULTS_DIR /
           f"queries__{args.dataset}__{condition.name}__{flatten_id(generator_config.id)}.json")
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "condition": condition.name,
        "prompt_version": condition.prompt_version,
        "prompt_sha": condition.prompt_sha,
        "generator": generator_config.id,
        "lineage": generator_config.lineage,
        "model_versions": sorted(versions),
        "temperature": experiment.temperature,
        "seed": experiment.seed,
        "skipped": skipped,
        "queries": transformed,
    }, indent=2))

    print(f"cache hits {hits}/{len(qids)} | model version(s): {sorted(versions)}")
    print(f"wrote {out}")

    sample_qid = next(iter(transformed), None)
    if sample_qid is not None:
        original = queries[sample_qid]
        print(f"\nexample\n  original ({len(original)} chars): {original[:220]}")
        print(f"  rewrite  ({len(transformed[sample_qid])} chars): {transformed[sample_qid][:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
