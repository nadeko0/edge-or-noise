"""Baseline scan: does a classifier rank 1-minute bars by future
scalp-trade profitability better than chance, and does that hold when
you vary stop-loss, direction, and prediction horizon?

Reproduces the headline finding: AUC ~0.58-0.67 depending on horizon,
consistently above the 0.50 no-signal baseline, but the unconditional
("trade every signal") profit factor stays well below 1.0 throughout --
see reports/FINAL_REPORT.md section 1-2 for the full numbers.
"""
from __future__ import annotations

from common import FEES, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, walk_forward


def labeled(df, pt=PT, sl=SL0, max_bars=MAX_BARS, is_long=True):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        pt, sl, max_bars, FEES, is_long,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def main():
    btc = get_live("BTCUSDT")

    print("\n== baseline: PT=0.20%, no SL, 5-bar horizon ==")
    r = walk_forward(labeled(btc), FEATURE_COLS_FULL, max_bars=MAX_BARS)
    print(r)

    print("\n== stop-loss variants ==")
    for sl, tag in [(0.0010, "SL=0.10%"), (0.0015, "SL=0.15%"), (0.0020, "SL=0.20% (symmetric)")]:
        r = walk_forward(labeled(btc, sl=sl), FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)
        print(f"{tag}: mean_auc={r['mean_auc']:.4f}  mean_base_pf={r['mean_base_pf']:.3f}")

    print("\n== short side ==")
    r = walk_forward(labeled(btc, is_long=False), FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)
    print(f"short: mean_auc={r['mean_auc']:.4f}  mean_base_pf={r['mean_base_pf']:.3f}")

    print("\n== horizon sweep ==")
    for mb in (1, 2, 3, 5, 10):
        r = walk_forward(labeled(btc, max_bars=mb), FEATURE_COLS_FULL, max_bars=mb, verbose=False)
        print(f"MaxBars={mb:2d}: mean_auc={r['mean_auc']:.4f}  mean_base_pf={r['mean_base_pf']:.3f}")


if __name__ == "__main__":
    main()
