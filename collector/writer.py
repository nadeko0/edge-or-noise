"""Daily rotating JSONL+gzip writer.

Creates one compressed file per symbol per day:
  data/trades/BTCUSDT_2026-04-22.jsonl.gz
  data/liquidations/all_2026-04-22.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DailyWriter:
    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir)
        # key -> (date_str, file_handle)
        self._handles: dict[str, tuple[str, Any]] = {}

    def _handle(self, stream: str, name: str):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{stream}/{name}"
        if key in self._handles:
            date_str, fh = self._handles[key]
            if date_str == today:
                return fh
            fh.close()  # day rollover
        path = self._base / stream / f"{name}_{today}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = gzip.open(path, "ab")  # binary mode -- avoids CPython 3.13 "lost gzip_file" bug
        self._handles[key] = (today, fh)
        return fh

    def _write(self, fh: Any, record: dict) -> None:
        fh.write((json.dumps(record, separators=(",", ":")) + "\n").encode())

    def write_trade(self, symbol: str, record: dict) -> None:
        self._write(self._handle("trades", symbol), record)

    def write_liq(self, record: dict) -> None:
        self._write(self._handle("liquidations", "all"), record)

    def write_ticker(self, symbol: str, record: dict) -> None:
        self._write(self._handle("tickers", symbol), record)

    def write_orderbook(self, symbol: str, record: dict) -> None:
        self._write(self._handle("orderbook", symbol), record)

    def write_orderbook_rpi(self, symbol: str, record: dict) -> None:
        self._write(self._handle("orderbook_rpi", symbol), record)

    def flush_all(self) -> None:
        for _, fh in self._handles.values():
            try:
                fh.flush()
            except Exception:
                pass

    def close_all(self) -> None:
        for _, fh in self._handles.values():
            fh.close()
        self._handles.clear()
