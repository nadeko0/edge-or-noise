"""Triple-barrier labeling for a fixed-horizon long/short scalp trade.

At bar i, a hypothetical trade enters at the next bar's open (i+1) and
is closed by whichever comes first:
  - price touches the take-profit level (pt_pct away from entry)
  - price touches the stop-loss level (sl_pct away from entry), if enabled
  - `max_bars` bars elapse with neither touched -> exit at that bar's close

This produces a label (1 = trade would have been net profitable after
fees, 0 = not) and the realized pnl (fraction of notional) for EVERY
bar, independent of any trading rule -- the label answers "would a
trade started here have worked", not "should a rule have fired here".
That separation is what lets you test whether a classifier can rank
bars by future profitability, rather than testing a single hand-picked
rule.

No lookahead: label[i] only uses open/high/low/close of bars > i.
"""
from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def label_triple_barrier(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pt_pct: float,
    sl_pct: float,
    max_bars: int,
    fees: float,
    is_long: bool = True,
):
    """Returns (label int8[-1/0/1], pnl float[nan or fraction]).

    sl_pct <= 0 disables the stop-loss (pure TP / time-exit).
    label == -1 marks bars with no valid future window (near the end
    of the series) -- filter these out before training/evaluating.
    """
    n = len(open_)
    label = np.full(n, -1, dtype=np.int8)
    pnl = np.full(n, np.nan)
    for i in range(n - 1):
        ie = i + 1
        if ie >= n:
            continue
        ep = open_[ie]
        if ep <= 0.0:
            continue
        if is_long:
            tp = ep * (1.0 + pt_pct)
            sl = ep * (1.0 - sl_pct) if sl_pct > 0 else -1.0
        else:
            tp = ep * (1.0 - pt_pct)
            sl = ep * (1.0 + sl_pct) if sl_pct > 0 else 1e18
        filled = False
        last_j = max_bars
        for j in range(max_bars):
            idx = ie + j
            if idx >= n:
                last_j = j
                break
            h, lo = high[idx], low[idx]
            if is_long:
                if sl_pct > 0 and lo <= sl:
                    pnl[i] = -sl_pct - fees
                    filled = True
                    break
                if h >= tp:
                    pnl[i] = pt_pct - fees
                    filled = True
                    break
            else:
                if sl_pct > 0 and h >= sl:
                    pnl[i] = -sl_pct - fees
                    filled = True
                    break
                if lo <= tp:
                    pnl[i] = pt_pct - fees
                    filled = True
                    break
        if not filled:
            il = min(ie + last_j - 1, n - 1)
            if il < ie:
                continue
            xp = close[il]
            gross = (xp - ep) / ep if is_long else (ep - xp) / ep
            pnl[i] = gross - fees
        label[i] = 1 if pnl[i] > 0 else 0
    return label, pnl


@numba.njit(cache=True)
def label_triple_barrier_with_mae(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pt_pct: float,
    max_bars: int,
    fees: float,
):
    """Long-only variant that additionally reports, per bar, the worst
    adverse excursion (MAE) reached before the natural exit -- used to
    simulate realistic liquidation risk at a given leverage instead of
    naively multiplying pnl by leverage."""
    n = len(open_)
    label = np.full(n, -1, dtype=np.int8)
    pnl = np.full(n, np.nan)
    mae_pct = np.full(n, np.nan)
    for i in range(n - 1):
        ie = i + 1
        if ie >= n:
            continue
        ep = open_[ie]
        if ep <= 0.0:
            continue
        tp = ep * (1.0 + pt_pct)
        filled = False
        last_j = max_bars
        worst_low = ep
        for j in range(max_bars):
            idx = ie + j
            if idx >= n:
                last_j = j
                break
            h, lo = high[idx], low[idx]
            if lo < worst_low:
                worst_low = lo
            if h >= tp:
                pnl[i] = pt_pct - fees
                filled = True
                last_j = j
                break
        if not filled:
            il = min(ie + last_j - 1, n - 1)
            if il < ie:
                continue
            xp = close[il]
            pnl[i] = (xp - ep) / ep - fees
        mae_pct[i] = max((ep - worst_low) / ep, 0.0)
        label[i] = 1 if pnl[i] > 0 else 0
    return label, pnl, mae_pct


def profit_factor(pnls: np.ndarray) -> float:
    """sum(wins) / -sum(losses). 999.0 if there are no losses, 0.0 if
    there are no wins (or no trades at all)."""
    pnls = np.asarray(pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    if len(losses) == 0:
        return 999.0
    if len(wins) == 0:
        return 0.0
    return float(wins.sum() / -losses.sum())
