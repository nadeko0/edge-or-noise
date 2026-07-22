"""Shared setup for every experiment script: symbol list, date range,
label parameters, and a cached dataset builder so re-running a script
doesn't re-parse gigabytes of gzip every time.

Requires TICKML_DATA_DIR to point at a directory shaped like
collector/'s output (see README.md) and TICKML_SFEREZ_DIR for the
independent OOS dataset (optional -- only needed for *_sferez* scripts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tickml.config import CACHE_DIR, SFEREZ_DIR
from tickml.features import add_core_features
from tickml.loaders_live import build_bar_dataset as build_live
from tickml.loaders_sferez import build_bar_dataset as build_sferez

PT = 0.0020       # 0.20% take-profit
SL0 = 0.0         # no stop-loss by default
MAX_BARS = 5      # 5-minute time exit
FEES = 0.00040    # 0.04% round-trip (maker/limit assumption)

LIVE_START, LIVE_END = "2026-04-22", "2026-07-20"
LIVE_DATES = [d.strftime("%Y-%m-%d") for d in pd.date_range(LIVE_START, LIVE_END)]


def get_live(symbol: str) -> pd.DataFrame:
    # Cache key includes the date range on purpose: if LIVE_START/LIVE_END
    # ever change, a stale cache from a different range must NOT be
    # silently reused under the new-looking run.
    cache = CACHE_DIR / f"live_{symbol}_{LIVE_START}_{LIVE_END}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_live(symbol, LIVE_DATES)
    df = add_core_features(df)
    df.to_parquet(cache)
    return df


def get_sferez(symbol3: str) -> pd.DataFrame:
    """symbol3: 'BTC' / 'ETH' / 'SOL' (matches the historical dataset's
    directory naming, base asset only, no USDT suffix)."""
    base = SFEREZ_DIR / symbol3
    dates = sorted(p.name for p in base.iterdir() if p.is_dir())
    # Cache key includes the actual first/last date found on disk so a
    # differently-dated archive can never be silently served from a
    # stale cache (same rationale as get_live above).
    cache = CACHE_DIR / f"sferez_{symbol3}_{dates[0]}_{dates[-1]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = build_sferez(base, dates)
    df = add_core_features(df)
    df.to_parquet(cache)
    return df
