"""Rank-agreement measures between leaderboards (§5.5).

Used two ways. In Phase 4 it compares the human-query leaderboard against each transformed one.
Right now it does something more basic but equally important: compares *our reproduction* against
RTEB's published ordering. Agreement there is a stronger correctness signal than the per-model
gate, because a systematic error that shifted every model equally would pass all the individual
gates while destroying the ordering — and the ordering is what RQ1 actually measures.

RBO is implemented here rather than taken from the `rbo` package, which pins numpy<2 and would
hold the whole project back. It is ~20 lines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from scipy import stats

Scores = Mapping[str, float]


def ranking(scores: Scores) -> list[str]:
    """Model ids best-first. Ties break by id so a run is reproducible (§9)."""
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _aligned(a: Scores, b: Scores) -> tuple[list[float], list[float], list[str]]:
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        raise ValueError(f"need >=2 models in common, got {len(shared)}")
    return [a[k] for k in shared], [b[k] for k in shared], shared


def kendall_tau(a: Scores, b: Scores) -> tuple[float, float]:
    """Kendall's tau-b and its p-value. Secondary metric to RBO (§5.5)."""
    xs, ys, _ = _aligned(a, b)
    result = stats.kendalltau(xs, ys)
    return float(result.statistic), float(result.pvalue)


def spearman(a: Scores, b: Scores) -> tuple[float, float]:
    xs, ys, _ = _aligned(a, b)
    result = stats.spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def rbo(left: Sequence[str], right: Sequence[str], p: float = 0.9) -> float:
    """Rank-biased overlap: top-weighted agreement between two rankings.

    The headline correlation metric (§5.5), because a leaderboard reshuffle at the top matters
    far more than one at the bottom, and tau weights all positions equally. `p` sets how
    top-heavy the weighting is; 0.9 puts roughly 86% of the weight on the first 10 positions.

    This is the extrapolated form for rankings of equal length, so it is defined on the whole
    list rather than only its common prefix.
    """
    if not left or not right:
        return 0.0

    depth = min(len(left), len(right))
    overlaps, seen_left, seen_right = [], set(), set()
    intersection = 0
    for d in range(depth):
        seen_left.add(left[d])
        seen_right.add(right[d])
        intersection += (left[d] in seen_right) + (right[d] in seen_left)
        intersection -= left[d] == right[d]  # counted twice when the same item lands both sides
        overlaps.append(intersection / (d + 1))

    # RBO_ext for equal-length lists:  (1-p) * sum_{d=1..k} p^(d-1) A_d  +  p^k * A_k
    # The trailing term extrapolates the unseen tail by assuming agreement continues at A_k.
    # Identical rankings must give exactly 1.0, which is what the unit test pins.
    weighted = sum(p**d * overlaps[d] for d in range(depth))
    return float((1 - p) * weighted + overlaps[-1] * p**depth)


def compare(ours: Scores, theirs: Scores, p: float = 0.9) -> dict[str, float]:
    """Full agreement summary between two score maps over the same models."""
    tau, tau_p = kendall_tau(ours, theirs)
    rho, rho_p = spearman(ours, theirs)
    shared = sorted(set(ours) & set(theirs))
    ours_r = [m for m in ranking(ours) if m in shared]
    theirs_r = [m for m in ranking(theirs) if m in shared]
    return {
        "n_models": len(shared),
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "spearman": rho,
        "spearman_p": rho_p,
        "rbo": rbo(ours_r, theirs_r, p=p),
        "exact_order_match": float(ours_r == theirs_r),
        "max_abs_delta": max(abs(ours[m] - theirs[m]) for m in shared),
    }
