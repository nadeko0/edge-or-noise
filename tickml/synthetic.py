"""Synthetic tick data in the live collector's file layout, so the whole
pipeline (loaders -> bars -> features -> labels -> validation) can be
run end to end without the real dataset, which cannot be shipped.

Two worlds, same generator:
  - vol_clustering=True: volatility switches between a calm and a
    volatile regime lasting hours (volatility clustering, the one
    property the real-data study found predictive).
  - vol_clustering=False: constant volatility, i.e. nothing to find.

A working pipeline should report AUC clearly above 0.5 on the first and
~0.5 on the second (experiments/00_synthetic_smoke.py, and
tests/test_synthetic_pipeline.py). Nothing here resembles real Bybit
order flow beyond the file shapes; it is a correctness check, not a
market model.

Files written (same names/fields as collector/writer.py):
  trades/<SYM>_<DATE>.jsonl.gz        {T,s,S,v,p}
  tickers/<SYM>_<DATE>.jsonl.gz       {T,bid,bsz,ask,asz,oi,fr}
  liquidations/all_<DATE>.jsonl.gz    {T,s,S,v,p}
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
SEC_MS = 1_000

# Per-minute return sd ~0.07%, close to BTC's typical 1-minute scale,
# so a 0.20% / 5-minute barrier is hit at a realistic rate. The
# clustered world switches between a calm and a volatile regime with
# the same average, each lasting ~2 hours on average.
MINUTE_SD = 0.0007
CALM_SD, VOLATILE_SD = 0.0003, 0.0011
MEAN_REGIME_MIN = 120


def _minute_sd(n_min: int, rng: np.random.Generator, vol_clustering: bool) -> np.ndarray:
    if not vol_clustering:
        return np.full(n_min, MINUTE_SD)
    switch = rng.random(n_min) < 1.0 / MEAN_REGIME_MIN
    volatile = (np.cumsum(switch) + int(rng.integers(0, 2))) % 2 == 1
    return np.where(volatile, VOLATILE_SD, CALM_SD)


def _write_jsonl_gz(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def write_synthetic_dataset(out_dir: Path, symbol: str = "BTCUSDT", start: str = "2026-01-01",
                            n_days: int = 5, vol_clustering: bool = True, seed: int = 0) -> list[str]:
    """Writes n_days of synthetic files under out_dir, returns the dates."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n_days)]
    n_min = n_days * 1440
    sd_min = _minute_sd(n_min, rng, vol_clustering)

    # 1-second price path; each second inherits its minute's volatility
    sd_sec = np.repeat(sd_min / np.sqrt(60.0), 60)
    log_px = np.log(60_000.0) + np.cumsum(sd_sec * rng.standard_normal(n_min * 60))
    px = np.exp(log_px)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    oi = 50_000.0 + np.cumsum(rng.normal(0.0, 5.0, n_min))

    for di, date in enumerate(dates):
        day_t0 = t0 + di * DAY_MS
        sec = np.arange(di * 86_400, (di + 1) * 86_400)

        # a trade in ~1 of 3 seconds, more often when volatility is high
        busy = sd_sec[sec] / MINUTE_SD * np.sqrt(60.0)
        fire = rng.random(len(sec)) < np.clip(0.33 * busy, 0.05, 0.95)
        idx = sec[fire]
        side = np.where(rng.random(len(idx)) < 0.5, "Buy", "Sell")
        size = np.round(rng.exponential(0.05, len(idx)) + 0.001, 3)
        _write_jsonl_gz(out_dir / "trades" / f"{symbol}_{date}.jsonl.gz", (
            {"T": day_t0 + int(i - sec[0]) * SEC_MS + int(rng.integers(0, 1000)), "s": symbol,
             "S": s, "v": f"{v:.3f}", "p": f"{px[i]:.1f}"}
            for i, s, v in zip(idx, side, size)
        ))

        # top-of-book + OI + funding every 10 seconds
        tick_sec = sec[::10]
        half_spread = 0.05
        _write_jsonl_gz(out_dir / "tickers" / f"{symbol}_{date}.jsonl.gz", (
            {"T": day_t0 + int(i - sec[0]) * SEC_MS, "bid": f"{px[i] - half_spread:.1f}",
             "bsz": f"{rng.exponential(2.0):.3f}", "ask": f"{px[i] + half_spread:.1f}",
             "asz": f"{rng.exponential(2.0):.3f}", "oi": f"{oi[i // 60]:.1f}", "fr": "0.0001"}
            for i in tick_sec
        ))

        # a few liquidations per hour at random seconds
        liq_idx = np.sort(rng.choice(sec, size=96, replace=False))
        _write_jsonl_gz(out_dir / "liquidations" / f"all_{date}.jsonl.gz", (
            {"T": day_t0 + int(i - sec[0]) * SEC_MS, "s": symbol,
             "S": "Buy" if rng.random() < 0.5 else "Sell",
             "v": f"{rng.exponential(0.5):.3f}", "p": f"{px[i]:.1f}"}
            for i in liq_idx
        ))
    return dates
