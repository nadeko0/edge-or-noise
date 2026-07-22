"""Validation-primitive tests: permutation_test's null distribution
construction, equity_curve's compounding/drawdown math, and an
end-to-end smoke test that walk_forward runs without error on a tiny
synthetic dataset (not a statistical claim -- just "the pipeline
doesn't crash and returns the expected shape")."""
import numpy as np
import pandas as pd

from tickml.validation import equity_curve, permutation_test, walk_forward
from tickml.labeling import profit_factor


def test_permutation_test_returns_none_for_too_few_signals():
    pnl = np.random.default_rng(0).normal(0, 0.01, 1000)
    mask = np.zeros(1000, dtype=bool)
    mask[:3] = True  # only 3 signals, below the n>=10 floor
    assert permutation_test(pnl, mask) is None


def test_permutation_test_null_median_tracks_baseline_pf():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.0002, 0.01, 5000)  # slightly positive baseline drift
    mask = rng.random(5000) < 0.05
    res = permutation_test(pnl, mask, n_perms=200)
    assert res is not None
    assert res["n"] == int(mask.sum())
    # null median should be in the same ballpark as unconditional PF, not wildly different
    base_pf = profit_factor(pnl)
    assert abs(res["null_median"] - base_pf) < 1.5


def test_equity_curve_compounds_and_tracks_drawdown():
    df = pd.DataFrame({
        "bucket": [0, 60_000, 120_000, 180_000],
        "proba": [0.9, 0.9, 0.9, 0.9],
        "trade_pnl": [0.10, -0.20, 0.10, 0.10],  # +10%, -20%, +10%, +10%
    })
    res = equity_curve(df, threshold=0.5, max_bars=1, bar_ms=60_000,
                        start_capital=1000.0, position_fraction=1.0)
    assert res["n_trades"] == 4
    expected_equity = 1000 * 1.10 * 0.80 * 1.10 * 1.10
    assert res["final_equity"] == round(expected_equity, 2)
    # peak was after trade 1 (1100), trough after trade 2 (880) -> drawdown = 220/1100
    assert res["max_drawdown_pct"] == round(220 / 1100 * 100, 2)


def test_equity_curve_skips_overlapping_signals():
    # two signals 1 minute apart but max_bars=5 -> the second must be skipped
    df = pd.DataFrame({
        "bucket": [0, 60_000, 300_000],
        "proba": [0.9, 0.9, 0.9],
        "trade_pnl": [0.10, 0.10, 0.10],
    })
    res = equity_curve(df, threshold=0.5, max_bars=5, bar_ms=60_000, start_capital=1000.0)
    assert res["n_trades"] == 2  # bar at 60_000 falls inside the first trade's hold window


def test_walk_forward_smoke():
    rng = np.random.default_rng(2)
    n = 3000
    features = ["f1", "f2"]
    df = pd.DataFrame({f: rng.normal(0, 1, n) for f in features})
    df["label"] = (rng.random(n) < 0.5).astype(int)
    df["trade_pnl"] = np.where(df["label"] == 1, 0.01, -0.01)
    result = walk_forward(df, features, n_folds=2, max_bars=5, verbose=False)
    assert result["n_folds_run"] == 2
    assert 0.0 <= result["mean_auc"] <= 1.0
