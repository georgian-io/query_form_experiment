"""Rank-agreement tests (§5.5).

RBO gets real tests because it is the headline metric and it is hand-rolled here (the `rbo`
package pins numpy<2). The first version of it returned 1.11 for identical rankings -- a bound
violation that only surfaced when it was run on real data, so the bounds are pinned explicitly.
"""

from __future__ import annotations

import pytest

from src.eval.correlation import compare, kendall_tau, ranking, rbo, spearman


def test_ranking_is_best_first_and_tie_stable():
    assert ranking({"a": 0.5, "b": 0.9, "c": 0.1}) == ["b", "a", "c"]
    # Equal scores break by id, so a rerun cannot silently reorder the board (§9).
    assert ranking({"z": 0.5, "a": 0.5}) == ["a", "z"]


def test_rbo_is_bounded():
    """It is a similarity in [0, 1]; the original implementation returned 1.11."""
    for left, right in [
        (["a", "b", "c", "d"], ["a", "b", "c", "d"]),
        (["a", "b", "c", "d"], ["d", "c", "b", "a"]),
        (["a", "b", "c", "d"], ["b", "a", "d", "c"]),
    ]:
        assert 0.0 <= rbo(left, right) <= 1.0


def test_identical_rankings_score_exactly_one():
    assert rbo(list("abcdefgh"), list("abcdefgh")) == pytest.approx(1.0)


def test_rbo_is_top_weighted():
    """A swap at the top must cost more than the same swap at the bottom -- the point of RBO."""
    base = list("abcdef")
    top_swap = ["b", "a", "c", "d", "e", "f"]
    bottom_swap = ["a", "b", "c", "d", "f", "e"]
    assert rbo(base, top_swap) < rbo(base, bottom_swap)


def test_rbo_disjoint_rankings_score_zero():
    assert rbo(["a", "b"], ["c", "d"]) == pytest.approx(0.0)


def test_rbo_p_controls_top_weighting():
    base, swapped = list("abcdef"), ["b", "a", "c", "d", "e", "f"]
    # Smaller p concentrates weight at the top, so the same top swap hurts more.
    assert rbo(base, swapped, p=0.5) < rbo(base, swapped, p=0.99)


def test_tau_and_spearman_agree_on_perfect_and_reversed_order():
    ours = {"a": 3.0, "b": 2.0, "c": 1.0}
    same = {"a": 0.9, "b": 0.5, "c": 0.1}
    reverse = {"a": 0.1, "b": 0.5, "c": 0.9}
    assert kendall_tau(ours, same)[0] == pytest.approx(1.0)
    assert kendall_tau(ours, reverse)[0] == pytest.approx(-1.0)
    assert spearman(ours, same)[0] == pytest.approx(1.0)


def test_compare_reports_exact_match_and_worst_delta():
    ours = {"a": 0.50, "b": 0.40, "c": 0.30}
    theirs = {"a": 0.51, "b": 0.40, "c": 0.28}
    stats = compare(ours, theirs)
    assert stats["exact_order_match"] == 1.0
    assert stats["kendall_tau"] == pytest.approx(1.0)
    assert stats["rbo"] == pytest.approx(1.0)
    assert stats["max_abs_delta"] == pytest.approx(0.02)


def test_compare_needs_overlapping_models():
    with pytest.raises(ValueError, match=">=2 models in common"):
        compare({"a": 1.0}, {"b": 1.0})
