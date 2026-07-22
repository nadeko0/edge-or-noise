"""WebSocket connection manager.

Opens persistent connections to Bybit's public linear WS:
  1. publicTrade.{symbol}     -- every trade tick (CVD)
  2. allLiquidation.{symbol}  -- forced liquidation events
  3. tickers.{symbol}         -- OI, funding rate, bid/ask imbalance (1-2s cadence)
  4. orderbook.50.{symbol}    -- 50-level L2 book deltas
  5. orderbook.rpi.{symbol}   -- retail price improvement book

All auto-reconnect on disconnect with exponential back-off cap.
Bybit heartbeat ({"op":"ping"}) sent every PING_SEC seconds.
"""

from __future__ import annotations

import asyncio
import json
import logging

import websockets
import websockets.exceptions

from .config import PING_SEC, RECONNECT_SEC, WS_PUBLIC
from .state import CollectorState
from .writer import DailyWriter

logger = logging.getLogger(__name__)


async def _ping_loop(ws, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(PING_SEC)
        if stop.is_set():
            break
        try:
            await ws.send(json.dumps({"op": "ping"}))
        except Exception:
            break


async def _stream(
    name: str,
    args: list[str],
    state: CollectorState,
    writer: DailyWriter,
) -> None:
    delay = RECONNECT_SEC
    while True:
        try:
            async with websockets.connect(
                WS_PUBLIC,
                ping_interval=None,   # we handle heartbeat manually
                open_timeout=15,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                logger.info(f"[{name}] connected -- {len(args)} topics")
                delay = RECONNECT_SEC  # reset back-off on success

                stop = asyncio.Event()
                ping_task = asyncio.create_task(_ping_loop(ws, stop))

                try:
                    async for raw in ws:
                        _dispatch(name, raw, state, writer)
                finally:
                    stop.set()
                    ping_task.cancel()

        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.WebSocketException,
            OSError,
            asyncio.TimeoutError,
        ) as exc:
            logger.warning(f"[{name}] disconnected: {exc!r} -- retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)  # exponential back-off, cap 60s
        except Exception as exc:
            logger.error(f"[{name}] unexpected error: {exc!r} -- retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


def _dispatch(
    name: str,
    raw: str,
    state: CollectorState,
    writer: DailyWriter,
) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return

    topic = msg.get("topic", "")
    data  = msg.get("data")

    if not topic or data is None:
        # ping/pong or subscription ack -- ignore
        return

    if topic.startswith("publicTrade."):
        sym = topic.split(".", 1)[1]
        for t in data:  # data is a list of trades
            record = {
                "T": t["T"],
                "s": sym,
                "S": t["S"],   # Buy / Sell
                "v": t["v"],   # size (string)
                "p": t["p"],   # price (string)
            }
            writer.write_trade(sym, record)
            state.on_trade(sym, int(t["T"]), t["S"], float(t["v"]))

    elif topic.startswith("allLiquidation."):
        sym = topic.split(".", 1)[1]
        items = data if isinstance(data, list) else [data]
        for liq in items:
            record = {
                "T": liq["T"],
                "s": sym,
                "S": liq["S"],   # Buy (long liq) / Sell (short liq)
                "v": liq["v"],
                "p": liq["p"],
            }
            writer.write_liq(record)
            state.on_liquidation(sym, record)
            logger.info(
                f"[liq] {sym:15s} {liq['S']:4s} "
                f"size={float(liq['v']):.4f} price={liq['p']}"
            )

    elif topic.startswith("orderbook.rpi."):
        # topic: "orderbook.rpi.BTCUSDT"
        # Each level: [price, normal_size, rpi_size]
        # normal_size = institutional/regular limit orders
        # rpi_size    = retail price improvement orders
        sym      = topic.split(".", 2)[2]
        msg_type = msg.get("type", "delta")
        ts_ms    = msg.get("cts") or msg.get("ts", 0)
        record = {
            "T":  ts_ms,
            "s":  sym,
            "tp": msg_type,
            "b":  data.get("b", []),   # [[price, normal_sz, rpi_sz], ...]
            "a":  data.get("a", []),
            "u":  data.get("u"),
        }
        writer.write_orderbook_rpi(sym, record)

    elif topic.startswith("orderbook."):
        # topic: "orderbook.50.BTCUSDT"
        parts    = topic.split(".", 2)
        sym      = parts[2] if len(parts) == 3 else topic.split(".")[-1]
        msg_type = msg.get("type", "delta")
        ts_ms    = msg.get("cts") or msg.get("ts", 0)  # cts = matching engine ts
        bids     = data.get("b", [])
        asks     = data.get("a", [])
        metrics  = state.on_orderbook(sym, msg_type, bids, asks, ts_ms)
        if metrics:
            writer.write_orderbook(sym, metrics)

    elif topic.startswith("tickers."):
        sym   = topic.split(".", 1)[1]
        ts_ms = msg.get("ts", 0)
        # Write every message raw: {T, s, tp, ...all fields bybit sent}
        # "tp" = "snapshot" (first msg, all fields) or "delta" (only changed fields)
        # Reconstruct full state locally by forward-filling deltas
        record = {"T": ts_ms, "s": sym, "tp": msg.get("type", "delta")}
        record.update(data)
        writer.write_ticker(sym, record)
        # Still update in-memory state for Redis live signals
        state.on_ticker(sym, data, ts_ms)


async def run_collector(state: CollectorState, writer: DailyWriter) -> None:
    from .config import SYMBOLS

    trade_args   = [f"publicTrade.{s}"      for s in SYMBOLS]
    liq_args     = [f"allLiquidation.{s}"  for s in SYMBOLS]
    ticker_args  = [f"tickers.{s}"         for s in SYMBOLS]
    ob_args      = [f"orderbook.50.{s}"    for s in SYMBOLS]
    ob_rpi_args  = [f"orderbook.rpi.{s}"   for s in SYMBOLS]

    await asyncio.gather(
        _stream("trades",       trade_args,   state, writer),
        _stream("liquidations", liq_args,     state, writer),
        _stream("tickers",      ticker_args,  state, writer),
        _stream("orderbook",    ob_args,      state, writer),
        _stream("ob_rpi",       ob_rpi_args,  state, writer),
    )
