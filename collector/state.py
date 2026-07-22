"""In-memory rolling state + Redis flush.

Keeps rolling trade buffer (deque) per symbol for CVD calculation.
Flushes snapshot to Redis every FLUSH_SEC seconds so a separate
strategy/execution process can read it without touching the collector.

Redis keys written:
  cvd:{SYMBOL}    -> hash  {cvd_1m, cvd_5m, trades_1m, ts_ms}
  liqs:{SYMBOL}   -> JSON string  [list of last LIQ_KEEP liquidation dicts]
  ticker:{SYMBOL} -> hash  {oi, fr, bid, bid_sz, ask, ask_sz, last, mark, spread, ts_ms}
  ob:{SYMBOL}     -> hash  {bid5, ask5, imb, mid, spd, ts_ms}
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque

from .config import CVD_WINDOW, LIQ_KEEP, REDIS_URL

logger = logging.getLogger(__name__)


class CollectorState:
    def __init__(self) -> None:
        self._redis = None
        # {symbol: deque of (ts_sec: float, delta: float)}
        self._trades: dict[str, deque] = defaultdict(deque)
        # {symbol: list of liq dicts}
        self._liqs: dict[str, list] = defaultdict(list)
        # {symbol: current merged ticker state}
        self._ticker: dict[str, dict] = {}
        # {symbol: last ticker disk-write timestamp}
        self._ticker_ts: dict[str, float] = {}
        # {symbol: {'bids': {price: size}, 'asks': {price: size}}}
        self._orderbook: dict[str, dict] = {}
        # {symbol: last orderbook disk-write timestamp}
        self._ob_ts: dict[str, float] = {}

    async def connect(self) -> bool:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await self._redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) -- running without Redis")
            self._redis = None
            return False

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # -- Event handlers --------------------------------------------------

    def on_trade(self, symbol: str, ts_ms: int, side: str, size: float) -> None:
        ts = ts_ms / 1000
        delta = size if side == "Buy" else -size
        buf = self._trades[symbol]
        buf.append((ts, delta))
        cutoff = time.time() - CVD_WINDOW
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def on_liquidation(self, symbol: str, record: dict) -> None:
        lst = self._liqs[symbol]
        lst.append(record)
        if len(lst) > LIQ_KEEP:
            self._liqs[symbol] = lst[-LIQ_KEEP:]

    def on_ticker(self, symbol: str, data: dict, ts_ms: int) -> dict | None:
        """Merge delta/snapshot ticker; return complete state once per second."""
        state = self._ticker.setdefault(symbol, {})
        for k, v in data.items():
            if v is not None and v != "":
                state[k] = v
        state["T"] = ts_ms
        now = time.time()
        if now - self._ticker_ts.get(symbol, 0) >= 1.0:
            self._ticker_ts[symbol] = now
            return dict(state)
        return None

    def on_orderbook(
        self,
        symbol: str,
        msg_type: str,
        bids: list,
        asks: list,
        ts_ms: int,
    ) -> dict | None:
        """Apply snapshot or delta update; return imbalance metrics once per second."""
        ob = self._orderbook.setdefault(symbol, {"bids": {}, "asks": {}})

        if msg_type == "snapshot":
            ob["bids"] = {p: s for p, s in bids}
            ob["asks"] = {p: s for p, s in asks}
        else:  # delta
            for p, s in bids:
                if s == "0":
                    ob["bids"].pop(p, None)
                else:
                    ob["bids"][p] = s
            for p, s in asks:
                if s == "0":
                    ob["asks"].pop(p, None)
                else:
                    ob["asks"][p] = s

        now = time.time()
        if now - self._ob_ts.get(symbol, 0) >= 0.5:
            self._ob_ts[symbol] = now
            return self._ob_snapshot(symbol, ts_ms)
        return None

    def _ob_snapshot(self, symbol: str, ts_ms: int) -> dict:
        ob = self._orderbook.get(symbol, {"bids": {}, "asks": {}})
        bids = sorted(ob["bids"].items(), key=lambda x: float(x[0]), reverse=True)
        asks = sorted(ob["asks"].items(), key=lambda x: float(x[0]))
        return {
            "T": ts_ms,
            "s": symbol,
            "b": [[p, s] for p, s in bids],   # all bid levels, best first
            "a": [[p, s] for p, s in asks],   # all ask levels, best first
        }

    # -- Public reads -----------------------------------------------------

    def cvd(self, symbol: str, window_sec: int = CVD_WINDOW) -> float:
        cutoff = time.time() - window_sec
        return sum(d for ts, d in self._trades[symbol] if ts >= cutoff)

    def recent_liqs(self, symbol: str, window_sec: int = 30) -> list:
        cutoff = time.time() - window_sec
        return [liq for liq in self._liqs[symbol] if liq["T"] / 1000 >= cutoff]

    def get_ticker(self, symbol: str) -> dict:
        return dict(self._ticker.get(symbol, {}))

    def get_ob_snapshot(self, symbol: str) -> dict:
        return self._ob_snapshot(symbol, int(time.time() * 1000))

    def get_ob_metrics(self, symbol: str) -> dict:
        return self.get_ob_snapshot(symbol)

    # -- Redis flush --------------------------------------------------------

    async def flush_to_redis(self) -> None:
        if not self._redis:
            return
        now = time.time()
        try:
            pipe = self._redis.pipeline()

            for sym, buf in self._trades.items():
                cvd_1m    = sum(d for ts, d in buf if ts >= now - 60)
                cvd_5m    = sum(d for ts, d in buf if ts >= now - 300)
                trades_1m = sum(1 for ts, _ in buf if ts >= now - 60)
                pipe.hset(f"cvd:{sym}", mapping={
                    "cvd_1m":    round(cvd_1m, 4),
                    "cvd_5m":    round(cvd_5m, 4),
                    "trades_1m": trades_1m,
                    "ts_ms":     int(now * 1000),
                })

            for sym, liqs in self._liqs.items():
                if liqs:
                    pipe.set(f"liqs:{sym}", json.dumps(liqs[-LIQ_KEEP:]))

            for sym, tick in self._ticker.items():
                bid = tick.get("bid1Price")
                ask = tick.get("ask1Price")
                spread = (
                    round((float(ask) - float(bid)) / float(bid) * 100, 6)
                    if bid and ask and float(bid) > 0 else None
                )
                mapping = {
                    k: v for k, v in {
                        "oi":     tick.get("openInterest"),
                        "fr":     tick.get("fundingRate"),
                        "bid":    bid,
                        "bid_sz": tick.get("bid1Size"),
                        "ask":    ask,
                        "ask_sz": tick.get("ask1Size"),
                        "last":   tick.get("lastPrice"),
                        "mark":   tick.get("markPrice"),
                        "spread": spread,
                        "ts_ms":  tick.get("T"),
                    }.items() if v is not None
                }
                if mapping:
                    pipe.hset(f"ticker:{sym}", mapping=mapping)

            for sym in self._orderbook:
                m = self._ob_snapshot(sym, int(now * 1000))
                pipe.hset(f"ob:{sym}", mapping={
                    "b":     json.dumps(m["b"][:10]),  # top 10 for live use
                    "a":     json.dumps(m["a"][:10]),
                    "ts_ms": m["T"],
                })

            await pipe.execute()
        except Exception as e:
            logger.debug(f"Redis flush error: {e}")
