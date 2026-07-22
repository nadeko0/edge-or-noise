"""Loader for the independent 2024 historical dataset used as the
out-of-sample regime check (a different year, never touched during
training or hyperparameter/threshold selection).

This dataset was NOT collected by this project's own WS collector --
it is a third-party historical export found online, covering BTC/ETH/SOL
only, structured as one zip file per (kind, symbol, day):

  <root>/<SYMBOL>/<DATE>/trades_<SYMBOL>_<DATE>.zip
  <root>/<SYMBOL>/<DATE>/tickers_<SYMBOL>_<DATE>.zip
  <root>/<SYMBOL>/<DATE>/liquidations_<SYMBOL>_<DATE>.zip

Each zip contains one ndjson file where every line is a 1-second
envelope {"t": ms, "d": <payload>} -- a list of trades for trades/liqs,
or a single ticker-snapshot dict for tickers. Not shipped with this
repo; point TICKML_SFEREZ_DIR at your own copy if you have one.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import polars as pl

from .config import BAR_MS

_LIQ_SCHEMA = {"t": pl.Int64, "d": pl.List(pl.Struct({
    "updatedTime": pl.Int64, "symbol": pl.Utf8, "side": pl.Utf8,
    "size": pl.Utf8, "price": pl.Utf8,
}))}


def _read_zip(path, schema=None):
    with zipfile.ZipFile(path) as z:
        data = z.read(z.namelist()[0])
    return pl.read_ndjson(io.BytesIO(data), schema=schema) if schema else pl.read_ndjson(io.BytesIO(data))


def load_trade_bars(base_dir, dates: list[str]) -> pl.DataFrame | None:
    frames = []
    for d in dates:
        p = base_dir / d / f"trades_{base_dir.name}_{d}.zip"
        if not p.exists():
            continue
        df = _read_zip(p).explode("d").unnest("d")
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


def load_liquidation_bars(base_dir, dates: list[str]) -> pl.DataFrame:
    frames = []
    for d in dates:
        p = base_dir / d / f"liquidations_{base_dir.name}_{d}.zip"
        if not p.exists():
            continue
        df = _read_zip(p, schema=_LIQ_SCHEMA).explode("d")
        df = df.filter(pl.col("d").is_not_null()).unnest("d")
        if df.height == 0:
            continue
        df = df.with_columns([
            pl.col("size").cast(pl.Float64),
            ((pl.col("updatedTime") // BAR_MS) * BAR_MS).alias("bucket"),
        ]).with_columns([
            pl.when(pl.col("side") == "Buy").then(pl.col("size")).otherwise(0.0).alias("liq_buy_v"),
            pl.when(pl.col("side") == "Sell").then(pl.col("size")).otherwise(0.0).alias("liq_sell_v"),
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


def load_ticker_bars(base_dir, dates: list[str]) -> pl.DataFrame:
    frames = []
    for d in dates:
        p = base_dir / d / f"tickers_{base_dir.name}_{d}.zip"
        if not p.exists():
            continue
        df = _read_zip(p).unnest("d")
        df = df.with_columns([
            pl.col("bid1Price").cast(pl.Float64, strict=False), pl.col("bid1Size").cast(pl.Float64, strict=False),
            pl.col("ask1Price").cast(pl.Float64, strict=False), pl.col("ask1Size").cast(pl.Float64, strict=False),
            pl.col("openInterest").cast(pl.Float64, strict=False), pl.col("fundingRate").cast(pl.Float64, strict=False),
        ]).drop_nulls(["bid1Price", "ask1Price"])
        if df.height == 0:
            continue
        df = df.with_columns([
            ((pl.col("bid1Size") - pl.col("ask1Size")) / (pl.col("bid1Size") + pl.col("ask1Size") + 1e-9)).alias("obi"),
            ((pl.col("ask1Price") - pl.col("bid1Price")) / ((pl.col("ask1Price") + pl.col("bid1Price")) / 2.0)).alias("spread"),
            ((pl.col("t") // BAR_MS) * BAR_MS).alias("bucket"),
        ])
        g = df.group_by("bucket", maintain_order=True).agg([
            pl.col("obi").last().alias("obi"), pl.col("spread").last().alias("spread"),
            pl.col("openInterest").last().alias("oi"), pl.col("fundingRate").last().alias("funding"),
        ])
        frames.append(g)
    if not frames:
        return pl.DataFrame(schema={"bucket": pl.Int64, "obi": pl.Float64, "spread": pl.Float64,
                                     "oi": pl.Float64, "funding": pl.Float64})
    return pl.concat(frames).sort("bucket")


def build_bar_dataset(base_dir, dates: list[str]) -> pd.DataFrame:
    trade_bars = load_trade_bars(base_dir, dates)
    if trade_bars is None:
        raise RuntimeError(f"no trade data found under {base_dir} for the given dates")
    liq_bars = load_liquidation_bars(base_dir, dates)
    tick_bars = load_ticker_bars(base_dir, dates)

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
