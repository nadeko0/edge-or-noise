"""Real equity-curve simulation (compounding, one position at a time,
sequential) with an honest leverage/liquidation model: a position is
liquidated once the worst intrabar adverse move (MAE) since entry
reaches ~1/leverage (minus a maintenance-margin buffer), whichever
comes first relative to the trade's natural exit. This is NOT the same
as "multiply pnl by leverage" -- it can cap losses at -100% of margin
even when the natural exit would have been smaller.
"""
from __future__ import annotations

import numpy as np

from common import FEES, MAX_BARS, PT, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier_with_mae
from tickml.config import BAR_MS
from tickml.validation import walk_forward_predictions, equity_curve

MMR = 0.005  # maintenance-margin buffer, rough approximation
LEVERAGES = [1, 2, 3, 5, 10, 20]


def apply_leverage(df, leverage, mmr=MMR):
    liq_threshold = 1.0 / leverage - mmr
    d = df.copy()
    liquidated = d["mae_pct"] >= liq_threshold
    d["liquidated"] = liquidated
    d["trade_pnl_leveraged"] = np.where(liquidated, -1.0, d["trade_pnl"] * leverage)
    return d


def main():
    btc = get_live("BTCUSDT")
    label, pnl, mae = label_triple_barrier_with_mae(
        btc["open"].values.astype("float64"), btc["high"].values.astype("float64"),
        btc["low"].values.astype("float64"), btc["close"].values.astype("float64"),
        PT, MAX_BARS, FEES,
    )
    btc["label"], btc["trade_pnl"], btc["mae_pct"] = label, pnl, mae

    print(f"worst historical MAE over the whole dataset: {btc['mae_pct'].max() * 100:.2f}%")
    print(f"(liquidation threshold at 20x leverage: ~{100 / 20 - MMR * 100:.1f}%)\n")

    pred = walk_forward_predictions(btc, FEATURE_COLS_FULL, max_bars=MAX_BARS)

    for thr in (0.50, 0.55, 0.60):
        mask = pred["proba"] >= thr
        n_sig = int(mask.sum())
        if n_sig < 10:
            continue
        sub = pred[mask]
        print(f"-- threshold={thr}  n_signals={n_sig} --")
        for lev in LEVERAGES:
            lsub = apply_leverage(sub, lev)
            liq_rate = float(lsub["liquidated"].mean() * 100)
            eq = equity_curve(
                lsub.assign(trade_pnl=lsub["trade_pnl_leveraged"]),
                0.0, MAX_BARS, BAR_MS, start_capital=1000.0, position_fraction=1.0,
            )
            print(f"  {lev:>3}x: liq_rate={liq_rate:5.1f}%  final_eq=${eq['final_equity']:>9.2f}  "
                  f"ret={eq['total_return_pct']:>7.2f}%  maxDD={eq['max_drawdown_pct']:>6.2f}%  "
                  f"WR={eq['win_rate_pct']}%  PF={eq['pf']}")


if __name__ == "__main__":
    main()
