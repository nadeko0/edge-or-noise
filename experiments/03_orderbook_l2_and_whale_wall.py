"""Does the real L2 order book (replayed from raw 50-level deltas, not
just the top-of-book already in FEATURE_COLS_FULL) add anything?
Includes a literal 'whale wall' feature: an anomalously large single
resting order relative to its neighbors near the touch.
"""
from __future__ import annotations

import pandas as pd

from common import FEES, MAX_BARS, PT, SL0, LIVE_DATES, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, walk_forward
from tickml.config import CACHE_DIR, DATA_DIR
from tickml.orderbook_l2 import build_l2_bars


def main():
    btc = get_live("BTCUSDT")

    l2_cache = CACHE_DIR / "l2book_BTCUSDT.parquet"
    if l2_cache.exists():
        l2 = pd.read_parquet(l2_cache)
    else:
        l2 = build_l2_bars(DATA_DIR, "BTCUSDT", LIVE_DATES)
        l2.to_parquet(l2_cache)

    depth_cols = ["top5_bid", "top5_ask", "top10_bid", "top10_ask",
                  "deep_obi5", "deep_obi10", "book_depth_ratio",
                  "whale_bid_ratio", "whale_bid_dist", "whale_ask_ratio", "whale_ask_dist"]
    merged = btc.merge(l2[["bucket"] + depth_cols], on="bucket", how="left")
    for col in depth_cols:
        merged[col] = merged[col].ffill()

    label, pnl = label_triple_barrier(
        merged["open"].values.astype("float64"), merged["high"].values.astype("float64"),
        merged["low"].values.astype("float64"), merged["close"].values.astype("float64"),
        PT, SL0, MAX_BARS, FEES, True,
    )
    merged["label"], merged["trade_pnl"] = label, pnl

    r_base = walk_forward(merged, FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)
    r_top5 = walk_forward(merged, FEATURE_COLS_FULL + ["deep_obi5"], max_bars=MAX_BARS, verbose=False)
    r_whale = walk_forward(merged, FEATURE_COLS_FULL + ["whale_bid_ratio", "whale_ask_ratio"],
                            max_bars=MAX_BARS, verbose=False)

    print(f"top-of-book only (baseline): AUC={r_base['mean_auc']:.4f}")
    print(f"+ top5 book depth/imbalance: AUC={r_top5['mean_auc']:.4f}")
    print(f"+ whale-wall size ratio:     AUC={r_whale['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
