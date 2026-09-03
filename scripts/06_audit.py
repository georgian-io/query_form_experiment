"""§5.6 intent-drift audit runner.

Samples queries for a (dataset, condition), asks a cross-lineage judge whether each generator's
rewrite preserved the information need, and reports the per-generator and pooled drift rate against
the §1.3 threshold. Verdicts are cached, so a rerun is free and reproducible.

    uv run python scripts/06_audit.py --dataset chatdoctor --condition terse --sample 120

Drift depends only on the (original, transformed) query pair, so this needs no retrieval results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.drift import (  # noqa: E402
    DRIFT_THRESHOLD,
    IntentDriftAuditor,
    drift_rate,
    sample_qids,
)
from src.config import load_datasets, load_experiment  # noqa: E402
from src.data import load_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_GENERATORS = [
    "gemini-2.5-pro", "claude-opus-4-8", "gpt-5.1-2025-11-13", "meta-llama/llama-3.3-70b-instruct",
]


def _load_transformed(dataset: str, condition: str, generator: str) -> dict[str, str]:
    f = RESULTS_DIR / f"queries__{dataset}__{condition}__{generator.replace('/', '__')}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())["queries"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--sample", type=int, default=120,
                        help="queries to audit per condition (all of them if the set is smaller)")
    parser.add_argument("--generators", nargs="*", default=DEFAULT_GENERATORS)
    parser.add_argument("--limit", type=int, default=None, help="cap judgments, for smoke tests")
    args = parser.parse_args()

    experiment = load_experiment()
    dataset_config = load_datasets().dataset(args.dataset)
    dataset = load_dataset(args.dataset, dataset_config)
    originals = dataset.queries()

    by_id = {g.id: g for g in experiment.generators}
    # Judges are generator configs too (gemini/claude), reused across the run.
    judges = {jid: by_id[jid] for jid in {"gemini-2.5-pro", "claude-opus-4-8"} if jid in by_id}
    auditor = IntentDriftAuditor(judges, experiment.temperature, experiment.seed)

    qids = sample_qids(list(originals), args.sample, experiment.seed)
    print(f"{args.dataset}/{args.condition}: auditing {len(qids)} queries × "
          f"{len(args.generators)} generators | rubric intent_drift.{auditor.rubric_version} "
          f"(sha {auditor.rubric_sha}) | threshold {DRIFT_THRESHOLD:.0%}")

    per_generator: dict[str, dict] = {}
    all_verdicts = []
    judged = 0
    for generator in args.generators:
        lineage = by_id[generator].lineage
        transformed = _load_transformed(args.dataset, args.condition, generator)
        if not transformed:
            print(f"  WARNING: no transformed queries for {generator}; skipping")
            continue
        verdicts = []
        for qid in qids:
            if qid not in transformed:
                continue
            if args.limit is not None and judged >= args.limit:
                break
            verdicts.append(auditor.audit_one(
                dataset=args.dataset, qid=qid, condition=args.condition, generator=generator,
                generator_lineage=lineage, original=originals[qid], transformed=transformed[qid],
            ))
            judged += 1
        all_verdicts.extend(verdicts)
        stats = drift_rate(verdicts)
        per_generator[generator] = {"judge": auditor.judge_id_for(lineage), **stats}
        flag = "OK" if stats["below_threshold"] else "OVER THRESHOLD"
        print(f"  {generator:38s} judge={auditor.judge_id_for(lineage):16s} "
              f"drift {stats['drift_rate']:.1%} ({stats['drifted']}/{stats['n']}, "
              f"{stats['uncertain']} uncertain)  [{flag}]")

    pooled = drift_rate(all_verdicts)
    print(f"  POOLED drift {pooled['drift_rate']:.1%} "
          f"({pooled['drifted']}/{pooled['n']}, {pooled['uncertain']} uncertain)  "
          f"[{'OK' if pooled['below_threshold'] else 'OVER THRESHOLD'}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"audit__{args.dataset}__{args.condition}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset, "condition": args.condition,
        "rubric_version": auditor.rubric_version, "rubric_sha": auditor.rubric_sha,
        "threshold": DRIFT_THRESHOLD, "n_sampled": len(qids),
        "per_generator": per_generator, "pooled": pooled,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
