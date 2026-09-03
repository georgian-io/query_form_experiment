"""ΔNDCG: transformed query sets against the human-query baseline (§5.5, Phase 2).

Reads the per-query scores both conditions already wrote to `results/` and pairs them. Nothing
is re-retrieved here -- this is pure analysis over runs that already happened, so it is cheap to
re-run as more conditions land.

Reports three things, in the order they should be read:
  1. per model x generator ΔNDCG, flagged against the measured noise floor
  2. group-level shift by model family -- §5.5's "collective shift" question
  3. cross-generator direction agreement -- §1.2's self-preference control

    uv run python scripts/04_evaluate.py --dataset aila_statutes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_experiment  # noqa: E402
from src.eval.attribution import RESOLUTION, compute_delta, family_shift  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load_runs(dataset: str, mode: str) -> tuple[dict, dict]:
    """Return (human_by_model, transformed_by (model, generator))."""
    human, transformed = {}, {}
    for path in RESULTS_DIR.glob(f"{mode}__{dataset}__*.json"):
        r = json.loads(path.read_text())
        if r.get("condition") == "human":
            human[r["model"]] = r["per_query"]
        else:
            transformed[(r["model"], r.get("generator"))] = r["per_query"]
    return human, transformed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", default="dense")
    args = parser.parse_args()

    human, transformed = _load_runs(args.dataset, args.mode)
    if not transformed:
        print(f"no transformed runs for {args.dataset}/{args.mode}; run 03 with --condition first")
        return 1

    families = {m.id: m.family for m in load_experiment().embedding_models}
    results = []
    for (model, generator), scores in sorted(transformed.items()):
        if model not in human:
            print(f"skipping {model}/{generator}: no human baseline run")
            continue
        results.append(compute_delta(
            human[model], scores, model=model, dataset=args.dataset, generator=generator
        ))

    print(f"\n{'=' * 100}\nΔNDCG@10 — {args.dataset} ({args.mode}), paraphrase vs human queries")
    print(f"resolution floor = {RESOLUTION} (below this: no detectable effect)\n{'=' * 100}")
    print(f"{'model':<40}{'generator':<22}{'human':>8}{'transf':>8}{'Δ':>10}  detail")
    for r in sorted(results, key=lambda r: (r.model, r.generator)):
        print(f"{r.model:<40}{r.generator[:21]:<22}{r.human:>8.4f}{r.transformed:>8.4f}"
              f"{r.delta:>+10.5f}  {r.summary().split('  ', 1)[1]}")

    print(f"\n{'-' * 100}\nGROUP-LEVEL SHIFT BY FAMILY (§5.5 collective shift)\n{'-' * 100}")
    for generator in sorted({r.generator for r in results}):
        subset = [r for r in results if r.generator == generator]
        print(f"\n  {generator}")
        for family, stat in family_shift(subset, families).items():
            flag = "" if stat["resolvable"] else "  (below floor)"
            print(f"    {family:<22} n={stat['n_models']:<3} mean Δ = {stat['mean_delta']:+.5f}"
                  f"  sd {stat['sd']:.5f}{flag}")

    print(f"\n{'-' * 100}\nCROSS-GENERATOR AGREEMENT (§1.2 self-preference control)\n{'-' * 100}")
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        by_model[r.model][r.generator] = r.delta
    disagree = 0
    print(f"{'model':<40}{'signs':<26}  verdict")
    for model, deltas in sorted(by_model.items()):
        signs = {g: ("+" if d > RESOLUTION else "-" if d < -RESOLUTION else "0")
                 for g, d in deltas.items()}
        distinct = {s for s in signs.values() if s != "0"}
        verdict = "consistent" if len(distinct) <= 1 else "DIRECTION FLIPS"
        disagree += len(distinct) > 1
        print(f"{model:<40}{''.join(signs[g] for g in sorted(signs)):<26}  {verdict}")
    print(f"\n  {disagree}/{len(by_model)} models disagree in direction across generators.")
    print("  Flips imply the effect is generator-specific -- closer to self-preference bias")
    print("  than to a property of LLM-written queries (§1.2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
