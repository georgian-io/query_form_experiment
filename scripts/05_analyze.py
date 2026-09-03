"""Assemble our leaderboard from `results/` and compare its ORDER to RTEB's published one.

The per-model gate checks each score in isolation, which a systematic error could survive: a bias
that shifted every model by the same amount would pass all sixteen gates and still destroy the
ranking. Since RQ1 is about *rank* changes, ordering agreement is the check that matters, and it
is the same machinery Phase 4 uses against transformed leaderboards.

    uv run python scripts/05_analyze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_datasets  # noqa: E402
from src.eval.correlation import compare, ranking  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(dataset: str, mode: str) -> dict[str, float]:
    scores = {}
    for path in RESULTS_DIR.glob(f"{mode}__{dataset}__*.json"):
        record = json.loads(path.read_text())
        if record.get("condition") == "human":
            scores[record["model"]] = record["score"]
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="dense")
    args = parser.parse_args()

    datasets = load_datasets()
    for name in datasets.datasets:
        ours = _load(name, args.mode)
        published = datasets.dataset(name).published_ndcg_at_10
        shared = sorted(set(ours) & set(published))
        if len(shared) < 2:
            print(f"\n{name}: only {len(shared)} model(s) with a result and a target; skipped")
            continue

        header = f"{name}  ({args.mode}, human queries, n={len(shared)} models)"
        print(f"\n{'=' * 78}\n{header}\n{'=' * 78}")
        print(f"{'#':>2}  {'model':<42}{'ours':>9}{'RTEB':>9}{'delta':>10}  {'their #':>7}")

        theirs_rank = {m: i + 1 for i, m in enumerate(ranking({m: published[m] for m in shared}))}
        for i, model in enumerate(ranking({m: ours[m] for m in shared}), start=1):
            moved = theirs_rank[model] - i
            flag = "" if moved == 0 else f"  ({moved:+d})"
            print(f"{i:>2}  {model:<42}{ours[model]:>9.5f}{published[model]:>9.5f}"
                  f"{ours[model] - published[model]:>+10.5f}  {theirs_rank[model]:>7}{flag}")

        stats = compare({m: ours[m] for m in shared}, {m: published[m] for m in shared})
        print(f"\n  Kendall tau = {stats['kendall_tau']:+.4f}  (p={stats['kendall_p']:.2e})")
        print(f"  Spearman    = {stats['spearman']:+.4f}  (p={stats['spearman_p']:.2e})")
        print(f"  RBO (p=0.9) = {stats['rbo']:.4f}   <- top-weighted, the headline metric")
        print(f"  exact order match: {bool(stats['exact_order_match'])}")
        print(f"  largest single-model delta: {stats['max_abs_delta']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
