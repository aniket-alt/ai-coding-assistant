"""Tests for the runtime assertions and the ledger behind the audit panel."""

import numpy as np
import pandas as pd
import pytest

from dsfs import leakage


@pytest.fixture(autouse=True)
def _fresh_ledger():
    leakage.reset_ledger()
    yield
    leakage.reset_ledger()


def _frame(index):
    rng = np.random.RandomState(0)
    y = pd.Series(rng.randn(len(index)), index=index)
    X = pd.DataFrame({"noise": rng.randn(len(index))}, index=index)
    return X, y


@pytest.mark.parametrize("index", [
    pd.RangeIndex(200),
    pd.date_range("2020-01-01", periods=200, freq="D"),
])
def test_planted_target_leak_is_caught_for_any_index(index):
    # Regression: with a DatetimeIndex the check used to align X against a
    # RangeIndex target, get NaN for every correlation, and pass silently.
    X, y = _frame(index)
    X["leak"] = y.to_numpy()
    with pytest.raises(leakage.LeakageError, match="leak"):
        leakage.assert_no_target_leak(X, y)
    entry = leakage.ledger()[0]
    assert entry["passed"] is False
    assert entry["params"]["max_correlation_feature"] == "leak"


def test_clean_features_pass_and_record_a_real_measurement():
    X, y = _frame(pd.date_range("2020-01-01", periods=200, freq="D"))
    leakage.assert_no_target_leak(X, y)
    entry = leakage.ledger()[0]
    assert entry["passed"] is True
    assert entry["params"]["features_checked"] == 1
    # A measured correlation, not the empty placeholder the old bug produced.
    assert entry["params"]["max_correlation_feature"] == "noise"


def test_length_mismatch_is_an_error_not_a_pass():
    X, y = _frame(pd.RangeIndex(200))
    with pytest.raises(ValueError):
        leakage.assert_no_target_leak(X, y.iloc[:150])


def test_overlapping_splits_fail():
    with pytest.raises(leakage.LeakageError):
        leakage.assert_disjoint([1, 2, 3], [3, 4])
    assert leakage.ledger()[0]["params"]["n_overlap"] == 1


def test_temporal_lookahead_and_embargo():
    early = pd.to_datetime(["2024-01-01", "2024-01-02"])
    late = pd.to_datetime(["2024-01-05"])
    leakage.assert_temporal_split(early, late, embargo="1D")
    with pytest.raises(leakage.LeakageError):
        leakage.assert_temporal_split(late, early)
    with pytest.raises(leakage.LeakageError, match="Embargo"):
        leakage.assert_temporal_split(early, late, embargo="10D")


def test_elapsed_durations_are_not_given_an_invented_date():
    train = pd.to_timedelta([0, 3600], unit="s")
    test = pd.to_timedelta([7200], unit="s")
    out = leakage.assert_temporal_split(train, test)
    assert "1970" not in out["train_end"]
    assert out["gap"] == "0 days 01:00:00"


def test_ledger_only_contains_checks_that_ran():
    assert leakage.ledger() == []
    leakage.assert_disjoint([1], [2])
    assert [e["name"] for e in leakage.ledger()] == ["assert_disjoint"]


def test_require_records_and_raises():
    leakage.require("custom_ok", True, "fine", checked=3)
    with pytest.raises(leakage.LeakageError, match="custom_bad"):
        leakage.require("custom_bad", False, "broken", checked=1)
    names = [(e["name"], e["passed"]) for e in leakage.ledger()]
    assert names == [("custom_ok", True), ("custom_bad", False)]
    assert leakage.ledger()[0]["params"] == {"checked": 3}
