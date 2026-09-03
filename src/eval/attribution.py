"""ΔNDCG between a human-query baseline and a transformed condition (§5.5).

Per-query rather than mean-only, for two reasons. The obvious one is that §5.5's attribution
regression needs per-query deltas as its dependent variable. The less obvious one is that a mean
alone cannot distinguish "every query got slightly worse" from "most were unchanged and three
collapsed" -- and those imply completely different things about whether a transform is safe.

Everything here is paired: the same query, the same corpus, the same qrels, the same model. That
pairing is what makes a small delta interpretable at all, since it removes the between-query
variance that dominates the raw NDCG spread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import mean, stdev

from scipy import stats

# Below this, a delta is indistinguishable from serving variance and metric granularity.
# Measured by scripts/00_noise_floor.py: structural 0.00061 (AILAStatutes) / 0.00234 (HumanEval),
# empirical up to 0.00287. See CLAUDE.md.
RESOLUTION = 0.003


@dataclass(frozen=True)
class DeltaResult:
    model: str
    dataset: str
    generator: str
    human: float
    transformed: float
    per_query: dict[str, float]

    @property
    def delta(self) -> float:
        return self.transformed - self.human

    @property
    def resolvable(self) -> bool:
        """Whether the effect is larger than the instrument can resolve."""
        return abs(self.delta) >= RESOLUTION

    @property
    def n_better(self) -> int:
        return sum(1 for d in self.per_query.values() if d > 0)

    @property
    def n_worse(self) -> int:
        return sum(1 for d in self.per_query.values() if d < 0)

    @property
    def n_unchanged(self) -> int:
        return sum(1 for d in self.per_query.values() if d == 0)

    def wilcoxon(self) -> tuple[float, float]:
        """Paired signed-rank test over per-query deltas.

        Non-parametric on purpose: per-query NDCG@10 is bounded, discrete, and heavily massed at
        0 and 1, so a t-test's normality assumption does not hold. Returns (statistic, p); p is
        1.0 when every delta is zero, which `scipy` cannot handle.
        """
        deltas = [d for d in self.per_query.values() if d != 0]
        if not deltas:
            return 0.0, 1.0
        result = stats.wilcoxon(deltas)
        return float(result.statistic), float(result.pvalue)

    def summary(self) -> str:
        verdict = "resolvable" if self.resolvable else f"BELOW FLOOR ({RESOLUTION})"
        _, p = self.wilcoxon()
        return (f"{self.delta:+.5f} ({verdict})  "
                f"better/worse/same {self.n_better}/{self.n_worse}/{self.n_unchanged}  p={p:.3f}")


def compute_delta(
    human: Mapping[str, float],
    transformed: Mapping[str, float],
    *,
    model: str,
    dataset: str,
    generator: str,
) -> DeltaResult:
    """Pair two per-query score maps into a ΔNDCG result.

    The transformed set may be a subsample of the human set (§ ChatDoctor's --sample run), so
    the pairing runs over the transformed qids and BOTH means are recomputed over exactly those
    qids. What is not allowed is a transformed qid absent from the human baseline -- that is a
    real misalignment, not a subsample, and it would compare unrelated queries.
    """
    orphans = set(transformed) - set(human)
    if orphans:
        raise ValueError(
            f"{model}/{dataset}/{generator}: {len(orphans)} transformed qids have no human "
            f"baseline (e.g. {sorted(orphans)[:3]}); cannot pair them"
        )
    qids = list(transformed)
    return DeltaResult(
        model=model,
        dataset=dataset,
        generator=generator,
        human=mean(human[q] for q in qids),
        transformed=mean(transformed.values()),
        per_query={qid: transformed[qid] - human[qid] for qid in qids},
    )


def family_shift(results: list[DeltaResult], families: Mapping[str, str]) -> dict[str, dict]:
    """Group-level ΔNDCG by model family (§5.5's headline plot).

    This is the *collective shift* half of the analysis: whether the LLM-backbone cluster moves
    away from the older bi-encoders and BM25 under transformed queries, as distinct from any
    reshuffle within the cluster. Cross-model aggregation is legitimate here -- unlike for τ/RBO,
    where it would confound dataset selection with the transform (see CLAUDE.md).
    """
    grouped: dict[str, list[float]] = {}
    for r in results:
        grouped.setdefault(families.get(r.model, "unknown"), []).append(r.delta)
    return {
        family: {
            "n_models": len(deltas),
            "mean_delta": mean(deltas),
            "sd": stdev(deltas) if len(deltas) > 1 else 0.0,
            "resolvable": abs(mean(deltas)) >= RESOLUTION,
        }
        for family, deltas in sorted(grouped.items())
    }
