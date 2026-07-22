"""Literal, quantified versions of 'smart money' trading concepts
(not chart-drawing -- an actual computed feature/rule run through the
same walk-forward + permutation-test framework), plus open interest
and funding rate as first-class hypotheses rather than incidental
features.

1. Liquidity sweep / stop hunt: price wicks below a recent N-bar swing
   low but closes back above it, with a coincident liquidation-volume
   spike (confirms stops actually triggered, not just a random wick).
2. OI/price quadrant: new longs / new shorts / short-covering /
   long-liquidation as a categorical feature.
3. Funding-rate extremes and "new shorts" (OI up, price down) as
   standalone contrarian rules, each checked with a permutation test.
"""
from __future__ import annotations

import numba
import numpy as np

from common import FEES, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, permutation_test, walk_forward


@numba.njit(cache=True)
def sweep_signal(high, low, close, liq_sell, swing_window, liq_threshold):
    n = len(close)
    sig = np.zeros(n, dtype=np.bool_)
    roll_low = np.full(n, np.nan)
    for i in range(swing_window, n):
        lo = low[i - swing_window]
        for k in range(i - swing_window + 1, i):
            if low[k] < lo:
                lo = low[k]
        roll_low[i] = lo
    for i in range(swing_window, n):
        if np.isnan(roll_low[i]) or np.isnan(liq_threshold[i]):
            continue
        pierced = low[i] < roll_low[i]
        closed_back = close[i] > roll_low[i]
        liq_spike = liq_sell[i] > liq_threshold[i]
        if pierced and closed_back and liq_spike:
            sig[i] = True
    return sig


def labeled(df):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        PT, SL0, MAX_BARS, FEES, True,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def main():
    btc = labeled(get_live("BTCUSDT"))

    print("== liquidity sweep / stop hunt ==")
    liq_thresh = (btc["liq_sell_vol"].rolling(120).mean() + 2.0 * btc["liq_sell_vol"].rolling(120).std()).values
    sig = sweep_signal(btc["high"].values.astype("float64"), btc["low"].values.astype("float64"),
                        btc["close"].values.astype("float64"), btc["liq_sell_vol"].values.astype("float64"),
                        60, liq_thresh.astype("float64"))
    valid = btc[btc["label"] >= 0].reset_index(drop=True)
    mask = sig[btc["label"].values >= 0]
    res = permutation_test(valid["trade_pnl"].values, mask)
    print(f"n_signals={int(mask.sum())}  result={res}")
    print("(check reports/FINAL_REPORT.md for why this specific result should NOT be")
    print(" taken at face value without a parameter-sensitivity + independent-OOS check)")

    print("\n== OI / price quadrant ==")
    d = btc.copy()
    oi_chg_5 = d["oi"].pct_change(5)
    price_chg_5 = d["close"].pct_change(5)
    quadrant = np.select(
        [
            (oi_chg_5 > 0) & (price_chg_5 > 0),   # new longs
            (oi_chg_5 > 0) & (price_chg_5 <= 0),  # new shorts
            (oi_chg_5 <= 0) & (price_chg_5 > 0),  # short covering
        ],
        [0, 1, 2],
        default=3,  # long liquidation / capitulation
    )
    d["oi_quadrant"] = np.where(oi_chg_5.notna() & price_chg_5.notna(), quadrant, np.nan)
    r_quad = walk_forward(d, FEATURE_COLS_FULL + ["oi_quadrant"], max_bars=MAX_BARS, verbose=False)
    print(f"AUC with OI/price quadrant feature: {r_quad['mean_auc']:.4f}")

    print("\n== 'new shorts' as contrarian long signal ==")
    new_shorts_mask = ((oi_chg_5 > oi_chg_5.rolling(120).quantile(0.90)) & (price_chg_5 < 0)).fillna(False).values
    new_shorts_mask = new_shorts_mask[btc["label"].values >= 0]
    res2 = permutation_test(valid["trade_pnl"].values, new_shorts_mask)
    print(f"result={res2}")

    print("\n== funding-rate extreme as contrarian signal ==")
    fr_hi = btc["funding"].rolling(1440).quantile(0.95)
    fr_extreme_mask = (btc["funding"] > fr_hi).fillna(False).values
    fr_extreme_mask = fr_extreme_mask[btc["label"].values >= 0]
    res3 = permutation_test(valid["trade_pnl"].values, fr_extreme_mask)
    print(f"result={res3}")


if __name__ == "__main__":
    main()
