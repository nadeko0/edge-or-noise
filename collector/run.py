"""Bybit WS Collector entry point.

Subscribes to publicTrade + allLiquidation + tickers + orderbook(.50/.rpi)
for all configured symbols. Writes compressed JSONL to data/ and
optionally flushes CVD/orderbook state to Redis for a separate
execution process to read.

Usage:
    uv run collector/run.py                          # all configured symbols
    uv run collector/run.py --symbols BTCUSDT ETHUSDT # test mode
    uv run collector/run.py --no-redis                # skip Redis (disk only)

Output:
    data/trades/BTCUSDT_2026-04-22.jsonl.gz
    data/liquidations/all_2026-04-22.jsonl.gz
    data/tickers/BTCUSDT_2026-04-22.jsonl.gz
    data/orderbook/BTCUSDT_2026-04-22.jsonl.gz
    data/orderbook_rpi/BTCUSDT_2026-04-22.jsonl.gz
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import collector.config as cfg
from collector.collector import run_collector
from collector.state import CollectorState
from collector.writer import DailyWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collector")


async def _flush_loop(state: CollectorState, writer: DailyWriter, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        await state.flush_to_redis()
        writer.flush_all()


async def _stats_loop(state: CollectorState, interval: int = 30) -> None:
    """Print a live snapshot every 30s so you can verify data is flowing.

    This is diagnostic-only and must never be allowed to take down the
    collector: it runs alongside the 5 live WS streams in the same
    asyncio.gather(), so an uncaught exception here would cancel all of
    them too."""
    await asyncio.sleep(interval)
    while True:
        try:
            lines = []
            for sym in sorted(cfg.SYMBOLS[:6]):  # show first 6 symbols
                c1   = state.cvd(sym, 60)
                c5   = state.cvd(sym, 300)
                liqs = len(state.recent_liqs(sym, 300))
                tick = state.get_ticker(sym)
                ob   = state.get_ob_metrics(sym)
                oi   = tick.get("openInterest", "?")
                fr   = tick.get("fundingRate",  "?")
                lines.append(
                    f"  {sym:16s} cvd1m={c1:+9.2f}  cvd5m={c5:+9.2f}  "
                    f"liqs={liqs}  oi={oi}  fr={fr}"
                )
            logger.info("-- snapshot --\n" + "\n".join(lines))
        except Exception as exc:
            logger.error(f"[stats] snapshot formatting error (non-fatal): {exc!r}")
        await asyncio.sleep(interval)


async def main(symbols: list[str] | None, use_redis: bool) -> None:
    if symbols:
        cfg.SYMBOLS = symbols
        logger.info(f"Test mode: {symbols}")

    logger.info(
        f"Starting collector | symbols={len(cfg.SYMBOLS)} | "
        f"data={cfg.DATA_DIR} | redis={'yes' if use_redis else 'no'}"
    )

    state  = CollectorState()
    writer = DailyWriter(cfg.DATA_DIR)

    if use_redis:
        ok = await state.connect()
        if ok:
            logger.info(f"Redis connected: {cfg.REDIS_URL}")
    else:
        logger.info("Redis disabled (--no-redis)")

    try:
        tasks = [
            run_collector(state, writer),
            _stats_loop(state),
            _flush_loop(state, writer, cfg.FLUSH_SEC),
        ]
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Flushing and closing files...")
        writer.flush_all()
        writer.close_all()
        await state.close()
        logger.info("Collector stopped.")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bybit WS collector")
    p.add_argument(
        "--symbols", nargs="+",
        help="Symbols to collect (default: all configured symbols)",
    )
    p.add_argument(
        "--no-redis", action="store_true",
        help="Disable Redis (write to disk only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    try:
        asyncio.run(main(args.symbols, use_redis=not args.no_redis))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
