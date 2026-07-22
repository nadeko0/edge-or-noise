"""Correctness tests for the triple-barrier labeler -- specifically
targeting the two failure modes that matter most in this kind of
research: lookahead leakage and off-by-one errors in the exit window.
"""
import numpy as np
import pytest

from tickml.labeling import label_triple_barrier, label_triple_barrier_with_mae, profit_factor


def _flat_series(n, price=100.0):
    return (np.full(n, price), np.full(n, price), np.full(n, price), np.full(n, price))


def test_take_profit_triggers_and_pnl_is_correct():
    # entry at bar 1's open (100), bar 2 spikes to 102 (2% up) -> TP hit
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 102.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    close = np.array([100.0, 100.0, 101.0, 100.0, 100.0])
    label, pnl = label_triple_barrier(open_, high, low, close, pt_pct=0.02, sl_pct=0.0,
                                       max_bars=3, fees=0.0004, is_long=True)
    assert label[0] == 1
    assert pnl[0] == pytest.approx(0.02 - 0.0004)


def test_stop_loss_triggers_before_take_profit():
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.5, 100.0, 100.0])
    low = np.array([100.0, 100.0, 98.0, 100.0, 100.0])  # -2% before any TP
    close = np.array([100.0, 100.0, 99.0, 100.0, 100.0])
    label, pnl = label_triple_barrier(open_, high, low, close, pt_pct=0.02, sl_pct=0.01,
                                       max_bars=3, fees=0.0004, is_long=True)
    assert label[0] == 0
    assert pnl[0] == pytest.approx(-0.01 - 0.0004)


def test_time_exit_when_neither_barrier_hit():
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 101.0])
    high = np.array([100.0, 100.5, 100.5, 100.5, 101.0])
    low = np.array([100.0, 99.8, 99.8, 99.8, 101.0])
    close = np.array([100.0, 100.2, 100.1, 100.3, 101.0])
    label, pnl = label_triple_barrier(open_, high, low, close, pt_pct=0.05, sl_pct=0.05,
                                       max_bars=3, fees=0.0, is_long=True)
    # entry at open_[1]=100.0, exits at close of bar 1+3-1=3 -> close[3]=100.3
    assert pnl[0] == pytest.approx((100.3 - 100.0) / 100.0)


def test_last_bars_have_no_valid_label():
    n = 10
    o, h, l, c = _flat_series(n)
    label, pnl = label_triple_barrier(o, h, l, c, pt_pct=0.01, sl_pct=0.0,
                                       max_bars=5, fees=0.0004, is_long=True)
    # the very last bar can never have a valid next-bar entry
    assert label[-1] == -1
    assert np.isnan(pnl[-1])


def test_short_side_is_mirrored():
    open_ = np.array([100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 100.0])
    low = np.array([100.0, 100.0, 98.0, 100.0])  # -2% -> profitable for a short
    close = np.array([100.0, 100.0, 99.0, 100.0])
    label, pnl = label_triple_barrier(open_, high, low, close, pt_pct=0.02, sl_pct=0.0,
                                       max_bars=2, fees=0.0004, is_long=False)
    assert label[0] == 1
    assert pnl[0] == pytest.approx(0.02 - 0.0004)


def test_mae_reflects_worst_intrabar_excursion_before_exit():
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 100.0, 100.0, 102.0, 100.0])
    low = np.array([100.0, 100.0, 97.0, 100.0, 100.0])  # -3% dip before the eventual TP
    close = np.array([100.0, 100.0, 98.0, 101.0, 100.0])
    label, pnl, mae = label_triple_barrier_with_mae(open_, high, low, close,
                                                     pt_pct=0.02, max_bars=3, fees=0.0004)
    assert mae[0] == pytest.approx(0.03, abs=1e-9)


def test_profit_factor_edge_cases():
    assert profit_factor(np.array([])) == 0.0
    assert profit_factor(np.array([0.01, 0.02, -0.005])) == pytest.approx((0.01 + 0.02) / 0.005)
    assert profit_factor(np.array([0.01, 0.02])) == 999.0   # no losses at all
    assert profit_factor(np.array([-0.01, -0.02])) == 0.0   # no wins at all
