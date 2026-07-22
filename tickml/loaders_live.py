"""Loaders for the live collector's output shape:
  data/trades/<SYMBOL>_<DATE>.jsonl.gz        {T,s,S,v,p}
  data/liquidations/all_<DATE>.jsonl.gz       {T,s,S,v,p}  (all symbols in one file)
  data/tickers/<SYMBOL>_<DATE>.jsonl.gz       {T,bid,bsz,ask,asz,oi,fr}

See collector/writer.py for exactly how these are produced.
Tolerant of a truncated/corrupt gzip stream (can happen on an ungraceful
collector restart mid-write) -- recovers whatever valid lines exist
before the corruption point rather than failing the whole day.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd
import polars as pl

from .config import BAR_MS, DATA_DIR


def _read_ndjson_gz_safe(path, schema):
    try:
        return pl.read_ndjson(path, schema=schema)
    except Exception:
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            try:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            except Exception:
                pass
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)


def load_trade_bars(sym: str, dates: list[str]) -> pl.DataFrame | None:
    frames = []
    for d in dates:
        p = DATA_DIR / "trades" / f"{sym}_{d}.jsonl.gz"
        if not p.exists():
            continue
        df = _read_ndjson_gz_safe(p, schema={"T": pl.Int64, "s": pl.Utf8, "S": pl.Utf8,
                                              "v": pl.Utf8, "p": pl.Utf8})
        if df.height == 0:
            continue
        df = df.with_columns([
            pl.col("v").cast(pl.Float64), pl.col("p").cast(pl.Float64),
            ((pl.col("T") // BAR_MS) * BAR_MS).alias("bucket"),
        ]).with_columns([
            pl.when(pl.col("S") == "Buy").then(pl.col("v")).otherwise(0.0).alias("buy_v"),
            pl.when(pl.col("S") == "Sell").then(pl.col("v")).otherwise(0.0).alias("sell_v"),
        ])
        g = df.group_by("bucket", maintain_order=True).agg([
            pl.col("p").first().alias("open"), pl.col("p").max().alias("high"),
            pl.col("p").min().alias("low"), pl.col("p").last().alias("close"),
            pl.col("v").sum().alias("volume"), pl.col("buy_v").sum().alias("buy_vol"),
            pl.col("sell_v").sum().alias("sell_vol"),
        ])
        frames.append(g)
    return pl.concat(frames).sort("bucket") if frames else None


def load_liquidation_bars(sym: str, dates: list[str]) -> pl.DataFrame:
    frames = []
    for d in dates:
        p = DATA_DIR / "liquidations" / f"all_{d}.jsonl.gz"
        if not p.exists():
            continue
        df = _read_ndjson_gz_safe(p, schema={"T": pl.Int64, "s": pl.Utf8, "S": pl.Utf8,
                                              "v": pl.Utf8, "p": pl.Utf8})
        df = df.filter(pl.col("s") == sym)
        if df.height == 0:
            continue
        df = df.with_columns([
            pl.col("v").cast(pl.Float64), ((pl.col("T") // BAR_MS) * BAR_MS).alias("bucket"),
        ]).with_columns([
            pl.when(pl.col("S") == "Buy").then(pl.col("v")).otherwise(0.0).alias("liq_buy_v"),
            pl.when(pl.col("S") == "Sell").then(pl.col("v")).otherwise(0.0).alias("liq_sell_v"),
        ])
        g = df.group_by("bucket", maintain_order=True).agg([
            pl.col("liq_buy_v").sum().alias("liq_buy_vol"),
            pl.col("liq_sell_v").sum().alias("liq_sell_vol"),
        ])
        frames.append(g)
    if not frames:
        return pl.DataFrame(schema={"bucket": pl.Int64, "liq_buy_vol": pl.Float64,
                                     "liq_sell_vol": pl.Float64})
    return pl.concat(frames).sort("bucket")


def load_ticker_bars(sym: str, dates: list[str]) -> pl.DataFrame:
    frames = []
    for d in dates:
        p = DATA_DIR / "tickers" / f"{sym}_{d}.jsonl.gz"
        if not p.exists():
            continue
        df = _read_ndjson_gz_safe(p, schema={"T": pl.Int64, "bid": pl.Utf8, "bsz": pl.Utf8,
                                              "ask": pl.Utf8, "asz": pl.Utf8, "oi": pl.Utf8,
                                              "fr": pl.Utf8}).sort("T")
        df = df.with_columns([
            pl.col("bid").cast(pl.Float64, strict=False), pl.col("bsz").cast(pl.Float64, strict=False),
            pl.col("ask").cast(pl.Float64, strict=False), pl.col("asz").cast(pl.Float64, strict=False),
            pl.col("oi").cast(pl.Float64, strict=False), pl.col("fr").cast(pl.Float64, strict=False),
        ]).drop_nulls(["bid", "ask"])
        if df.height == 0:
            continue
        df = df.with_columns([
            ((pl.col("bsz") - pl.col("asz")) / (pl.col("bsz") + pl.col("asz") + 1e-9)).alias("obi"),
            ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2.0)).alias("spread"),
            ((pl.col("T") // BAR_MS) * BAR_MS).alias("bucket"),
        ])
        g = df.group_by("bucket", maintain_order=True).agg([
            pl.col("obi").last().alias("obi"), pl.col("spread").last().alias("spread"),
            pl.col("oi").last().alias("oi"), pl.col("fr").last().alias("funding"),
        ])
        frames.append(g)
    if not frames:
        return pl.DataFrame(schema={"bucket": pl.Int64, "obi": pl.Float64, "spread": pl.Float64,
                                     "oi": pl.Float64, "funding": pl.Float64})
    return pl.concat(frames).sort("bucket")


def build_bar_dataset(sym: str, dates: list[str]) -> pd.DataFrame:
    """Joins trades/liquidations/tickers into one 1-minute bar table."""
    trade_bars = load_trade_bars(sym, dates)
    if trade_bars is None:
        raise RuntimeError(f"no trade data found for {sym} in the given date range")
    liq_bars = load_liquidation_bars(sym, dates)
    tick_bars = load_ticker_bars(sym, dates)

    df = trade_bars.join(liq_bars, on="bucket", how="left") \
                    .join(tick_bars, on="bucket", how="left") \
                    .sort("bucket").to_pandas()
    df["liq_buy_vol"] = df["liq_buy_vol"].fillna(0.0)
    df["liq_sell_vol"] = df["liq_sell_vol"].fillna(0.0)
    df["obi"] = df["obi"].ffill()
    df["spread"] = df["spread"].ffill()
    df["oi"] = df["oi"].ffill()
    df["funding"] = df["funding"].ffill()
    df["ts"] = pd.to_datetime(df["bucket"], unit="ms", utc=True)
    return df
