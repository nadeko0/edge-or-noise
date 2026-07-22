"""Feature engineering on 1-minute OHLCV(+orderflow) bars.

Every feature is a backward-looking rolling/lag computation over bars
<= i, so a bar's feature row is fully known at the moment a trade would
be entered (bar i+1's open) -- no lookahead into the labeling window.
"""
from __future__ import annotations

import pandas as pd


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Requires columns: open, high, low, close, volume, buy_vol,
    sell_vol, liq_buy_vol, liq_sell_vol, obi, spread, oi, funding, ts."""
    df = df.copy()
    df["cvd"] = df["buy_vol"] - df["sell_vol"]
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["cvd_sum_5"] = df["cvd"].rolling(5).sum()
    df["cvd_sum_20"] = df["cvd"].rolling(20).sum()
    df["vol_z20"] = (df["volume"] - df["volume"].rolling(20).mean()) / (
        df["volume"].rolling(20).std() + 1e-9
    )
    df["liq_buy_sum5"] = df["liq_buy_vol"].rolling(5).sum()
    df["liq_sell_sum5"] = df["liq_sell_vol"].rolling(5).sum()
    df["obi_ma5"] = df["obi"].rolling(5).mean()
    df["range_bar"] = (df["high"] - df["low"]) / df["open"]
    df["oi_chg_5"] = df["oi"].pct_change(5)
    df["hour"] = df["ts"].dt.hour
    df["is_london"] = df["hour"].between(8, 12).astype(int)
    return df


FEATURE_COLS_FULL = [
    "ret_1", "ret_5", "ret_15", "vol_5", "vol_20",
    "cvd", "cvd_sum_5", "cvd_sum_20", "vol_z20",
    "liq_buy_vol", "liq_sell_vol", "liq_buy_sum5", "liq_sell_sum5",
    "obi", "obi_ma5", "spread", "range_bar",
    "funding", "oi_chg_5",
    "hour", "is_london",
]

FEATURE_COLS_ORDERFLOW = [
    "cvd", "cvd_sum_5", "cvd_sum_20",
    "liq_buy_vol", "liq_sell_vol", "liq_buy_sum5", "liq_sell_sum5",
    "obi", "obi_ma5", "funding", "oi_chg_5", "hour", "is_london",
]

FEATURE_COLS_PRICEVOL = [
    "ret_1", "ret_5", "ret_15", "vol_5", "vol_20", "vol_z20",
    "spread", "range_bar", "hour", "is_london",
]
