"""ΔNDCG attribution tests (§5.5).

The pairing rules are the point: a paired delta is only meaningful if the two score maps line up
on the same queries, and the subsample path must pair over the transformed subset without
quietly changing the denominator.
"""

from __future__ import annotations

import pytest

from src.eval.attribution import RESOLUTION, compute_delta, family_shift


def _delta(human, transformed):
    return compute_delta(human, transformed, model="m", dataset="d", generator="g")


def test_paired_delta_over_identical_sets():
    r = _delta({"a": 0.4, "b": 0.6}, {"a": 0.5, "b": 0.5})
    assert r.per_query == {"a": pytest.approx(0.1), "b": pytest.approx(-0.1)}
    assert r.delta == pytest.approx(0.0)
    assert (r.n_better, r.n_worse, r.n_unchanged) == (1, 1, 0)


def test_subsample_pairs_over_the_transformed_subset():
    """A --sample run paraphrases only some queries; both means use exactly those."""
    human = {"a": 1.0, "b": 0.0, "c": 0.5}  # full baseline
    transformed = {"a": 0.8, "c": 0.9}  # subsample of 2
    r = _delta(human, transformed)
    assert set(r.per_query) == {"a", "c"}
    # Human mean is recomputed over {a, c} = 0.75, NOT the full-set 0.5 -- else the delta lies.
    assert r.human == pytest.approx(0.75)
    assert r.transformed == pytest.approx(0.85)
    assert r.delta == pytest.approx(0.10)


def test_transformed_qid_absent_from_human_is_an_error():
    """A subsample is a subset; a transformed qid with no baseline is misalignment, not sampling."""
    with pytest.raises(ValueError, match="no human baseline"):
        _delta({"a": 1.0}, {"a": 0.9, "z": 0.5})


def test_resolvable_tracks_the_noise_floor():
    below = _delta({"a": 0.5, "b": 0.5}, {"a": 0.5 + RESOLUTION / 2, "b": 0.5})
    above = _delta({"a": 0.5, "b": 0.5}, {"a": 0.5 + RESOLUTION * 2, "b": 0.5})
    assert not below.resolvable
    assert above.resolvable


def test_wilcoxon_is_defined_when_nothing_moved():
    r = _delta({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})
    assert r.wilcoxon() == (0.0, 1.0)


def test_family_shift_aggregates_by_model_family():
    results = [
        compute_delta({"q": 0.8}, {"q": 0.7}, model="big1", dataset="d", generator="g"),
        compute_delta({"q": 0.8}, {"q": 0.7}, model="big2", dataset="d", generator="g"),
        compute_delta({"q": 0.2}, {"q": 0.3}, model="small", dataset="d", generator="g"),
    ]
    families = {"big1": "llm-backbone", "big2": "llm-backbone", "small": "bi-encoder"}
    shift = family_shift(results, families)
    assert shift["llm-backbone"]["mean_delta"] == pytest.approx(-0.1)
    assert shift["llm-backbone"]["n_models"] == 2
    assert shift["bi-encoder"]["mean_delta"] == pytest.approx(0.1)
