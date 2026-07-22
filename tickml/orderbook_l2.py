"""Stateful L2 order-book replay from data/orderbook/ (raw 50-level
deltas), plus a 'whale wall' feature (an anomalously large single
resting order relative to its neighbors).

Two schemas exist in this project's own collected data, because the
collector's orderbook capture was upgraded partway through collection:
  - early days: a pre-aggregated {T, bid5, ask5, imb, mid, spd} snapshot
    (top-5-level size sums and imbalance already computed)
  - later days: raw 50-level deltas {T, b:[[price,size],...], a:[...]}
    that must be replayed into a running book

The raw-delta feed only sends updates within its own ~50-level window;
levels that scroll out of that window are never explicitly zeroed, so a
naive replay grows the in-memory book unboundedly. This implementation
periodically prunes to the nearest `keep_levels` price levels per side,
which is safe because top-N-nearest-touch features never need anything
beyond that.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BAR_MS


def build_l2_bars(data_dir: Path, sym: str, dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        p = data_dir / "orderbook" / f"{sym}_{d}.jsonl.gz"
        if not p.exists():
            continue
        t0 = time.time()
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                first = fh.readline()
            schema = "agg5" if '"bid5"' in first else "raw50"
        except Exception:
            continue
        day_rows = _load_agg5_day(p) if schema == "agg5" else _load_raw50_day(p)
        rows.extend(day_rows)
        print(f"  {d} [{schema}]: {len(day_rows)} bar snapshots  [{time.time() - t0:.1f}s]")
    return pd.DataFrame(rows).drop_duplicates(subset="bucket", keep="last").sort_values("bucket")


def _load_agg5_day(path):
    """Tolerant of a mid-file schema switch/truncation -- skips lines
    that don't match the expected keys instead of raising."""
    rows, cur_bucket, last = [], None, None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "bid5" not in o or "T" not in o:
                    continue
                bucket = (o["T"] // BAR_MS) * BAR_MS
                if cur_bucket is not None and bucket != cur_bucket and last is not None:
                    rows.append(_agg5_row(cur_bucket, last))
                cur_bucket = bucket
                last = o
    except Exception as e:
        print(f"    [WARN] corrupt/truncated agg5 orderbook, stopping day early: {path} ({e})")
    if last is not None:
        rows.append(_agg5_row(cur_bucket, last))
    return rows


def _agg5_row(bucket, o):
    return dict(bucket=bucket, top5_bid=o["bid5"], top5_ask=o["ask5"],
                top10_bid=np.nan, top10_ask=np.nan,
                deep_obi5=2.0 * o["imb"] - 1.0, deep_obi10=np.nan,
                book_depth_ratio=np.nan, n_bid_levels=np.nan, n_ask_levels=np.nan,
                whale_bid_ratio=np.nan, whale_bid_dist=np.nan,
                whale_ask_ratio=np.nan, whale_ask_dist=np.nan)


def _load_raw50_day(path, prune_every=2000, keep_levels=80):
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    cur_bucket, rows, n_msg = None, [], 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bucket = (o["T"] // BAR_MS) * BAR_MS
                if cur_bucket is not None and bucket != cur_bucket:
                    rows.append(_snapshot(cur_bucket, bids, asks))
                cur_bucket = bucket
                for pr, sz in o.get("b", []):
                    pr, sz = float(pr), float(sz)
                    if sz == 0.0:
                        bids.pop(pr, None)
                    else:
                        bids[pr] = sz
                for pr, sz in o.get("a", []):
                    pr, sz = float(pr), float(sz)
                    if sz == 0.0:
                        asks.pop(pr, None)
                    else:
                        asks[pr] = sz
                n_msg += 1
                if n_msg % prune_every == 0:
                    if len(bids) > keep_levels:
                        keep = sorted(bids.keys(), reverse=True)[:keep_levels]
                        bids = {p: bids[p] for p in keep}
                    if len(asks) > keep_levels:
                        keep = sorted(asks.keys())[:keep_levels]
                        asks = {p: asks[p] for p in keep}
    except Exception as e:
        print(f"    [WARN] corrupt/truncated orderbook, stopping day early: {path} ({e})")
    if cur_bucket is not None:
        rows.append(_snapshot(cur_bucket, bids, asks))
    return rows


def _whale_metrics(prices, sizes_dict, best_price, n_levels=20):
    """'Whale wall' = a single resting order much bigger than its
    neighbors near the touch. Returns (size-ratio-vs-local-median,
    pct distance from touch to that level)."""
    near = prices[:n_levels]
    if len(near) < 5:
        return np.nan, np.nan
    sizes = np.array([sizes_dict[p] for p in near])
    med = np.median(sizes)
    if med <= 0:
        return np.nan, np.nan
    imax = int(np.argmax(sizes))
    return float(sizes[imax] / med), abs(near[imax] - best_price) / best_price * 100.0


def _snapshot(bucket, bids, asks):
    if not bids or not asks:
        return dict(bucket=bucket, top5_bid=np.nan, top5_ask=np.nan, top10_bid=np.nan,
                    top10_ask=np.nan, deep_obi5=np.nan, deep_obi10=np.nan,
                    book_depth_ratio=np.nan, n_bid_levels=len(bids), n_ask_levels=len(asks),
                    whale_bid_ratio=np.nan, whale_bid_dist=np.nan,
                    whale_ask_ratio=np.nan, whale_ask_dist=np.nan)
    bid_prices = sorted(bids.keys(), reverse=True)
    ask_prices = sorted(asks.keys())
    top5_bid, top5_ask = sum(bids[p] for p in bid_prices[:5]), sum(asks[p] for p in ask_prices[:5])
    top10_bid, top10_ask = sum(bids[p] for p in bid_prices[:10]), sum(asks[p] for p in ask_prices[:10])
    deep_obi5 = (top5_bid - top5_ask) / (top5_bid + top5_ask + 1e-9)
    deep_obi10 = (top10_bid - top10_ask) / (top10_bid + top10_ask + 1e-9)
    book_depth_ratio = sum(bids.values()) / (sum(asks.values()) + 1e-9)
    whale_bid_ratio, whale_bid_dist = _whale_metrics(bid_prices, bids, bid_prices[0])
    whale_ask_ratio, whale_ask_dist = _whale_metrics(ask_prices, asks, ask_prices[0])
    return dict(bucket=bucket, top5_bid=top5_bid, top5_ask=top5_ask,
                top10_bid=top10_bid, top10_ask=top10_ask,
                deep_obi5=deep_obi5, deep_obi10=deep_obi10,
                book_depth_ratio=book_depth_ratio,
                n_bid_levels=len(bids), n_ask_levels=len(asks),
                whale_bid_ratio=whale_bid_ratio, whale_bid_dist=whale_bid_dist,
                whale_ask_ratio=whale_ask_ratio, whale_ask_dist=whale_ask_dist)
