"""Intent-drift audit tests (§5.6).

The verdict parsing and rate arithmetic are the parts that must not silently misclassify: a drift
audit that miscounts is worse than none, because it launders a qrel artifact into a "clean" number.
No network here -- the judge call is exercised behind a fixture in the integration suite.
"""

from __future__ import annotations

import pytest

from src.audit.drift import (
    DRIFT_THRESHOLD,
    JUDGE_FOR_LINEAGE,
    Verdict,
    _parse_verdict,
    drift_rate,
    sample_qids,
)


def _v(verdict: str) -> Verdict:
    return Verdict(verdict=verdict, reason="", judge="j", model_version="m", key={})


def test_parse_clean_json():
    v, r = _parse_verdict('{"verdict": "DRIFTED", "reason": "dropped the age constraint"}')
    assert v == "DRIFTED"
    assert "age" in r


def test_parse_json_embedded_in_prose():
    text = 'Sure —\n{"verdict":"PRESERVED","reason":"just reworded"}\nhope that helps'
    assert _parse_verdict(text)[0] == "PRESERVED"


def test_parse_bare_keyword_fallback():
    assert _parse_verdict("The rewrite has DRIFTED from the original.")[0] == "DRIFTED"


def test_parse_unparseable_raises():
    with pytest.raises(ValueError):
        _parse_verdict("I cannot help with that.")


def test_verdict_case_insensitive():
    assert _parse_verdict('{"verdict":"drifted","reason":"x"}')[0] == "DRIFTED"


def test_drift_rate_excludes_uncertain_from_denominator():
    # 2 drifted, 6 preserved, 2 uncertain -> 2 / (10 - 2) = 0.25
    verdicts = [_v("DRIFTED")] * 2 + [_v("PRESERVED")] * 6 + [_v("UNCERTAIN")] * 2
    r = drift_rate(verdicts)
    assert r["n"] == 10
    assert r["drifted"] == 2
    assert r["uncertain"] == 2
    assert r["drift_rate"] == pytest.approx(0.25)
    assert not r["below_threshold"]  # 0.25 >= 0.15


def test_drift_rate_below_threshold():
    verdicts = [_v("DRIFTED")] + [_v("PRESERVED")] * 19
    r = drift_rate(verdicts)
    assert r["drift_rate"] == pytest.approx(0.05)
    assert r["below_threshold"]


def test_drift_rate_all_uncertain_is_zero_not_error():
    r = drift_rate([_v("UNCERTAIN")] * 3)
    assert r["drift_rate"] == 0.0


def test_sample_is_seeded_and_deterministic():
    qids = [f"q{i}" for i in range(100)]
    a = sample_qids(qids, 20, seed=7)
    b = sample_qids(qids, 20, seed=7)
    assert a == b
    assert len(a) == 20
    assert set(a).issubset(set(qids))


def test_sample_returns_all_when_n_exceeds_population():
    qids = [f"q{i}" for i in range(10)]
    assert sample_qids(qids, 50, seed=1) == sorted(qids)


def test_every_generator_lineage_has_a_cross_lineage_judge():
    # The four generator lineages, each mapped to a judge of a DIFFERENT lineage.
    lineage_of_judge = {"gemini-2.5-pro": "google", "claude-opus-4-8": "anthropic"}
    for lineage, judge in JUDGE_FOR_LINEAGE.items():
        assert lineage_of_judge[judge] != lineage, f"{lineage} judged by its own lineage"


def test_threshold_is_pinned():
    assert DRIFT_THRESHOLD == 0.15


def test_judge_errors_counted_and_excluded_from_rate():
    err = Verdict(verdict="UNCERTAIN", reason="JUDGE_ERROR: ValueError: empty completion",
                  judge="j", model_version="JUDGE_ERROR::j", key={})
    verdicts = [_v("DRIFTED"), _v("PRESERVED"), _v("PRESERVED"), err]
    r = drift_rate(verdicts)
    assert r["judge_errors"] == 1
    assert r["uncertain"] == 1
    # rate is over the 3 judgeable verdicts, not 4: 1/3
    assert r["drift_rate"] == pytest.approx(1 / 3)
