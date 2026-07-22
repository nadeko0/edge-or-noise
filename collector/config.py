"""Collector configuration."""

from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../collector/

SYMBOLS = [
    "1000PEPEUSDT", "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT",
    "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DASHUSDT", "DOGEUSDT", "DOTUSDT",
    "ETCUSDT", "ETHUSDT", "LINKUSDT", "LTCUSDT", "MNTUSDT", "NEARUSDT",
    "OPUSDT", "POLUSDT", "SOLUSDT", "SUIUSDT", "TRXUSDT", "UNIUSDT",
    "WIFUSDT", "XRPUSDT",
]

WS_PUBLIC   = "wss://stream.bybit.com/v5/public/linear"
REDIS_URL   = "redis://localhost:6379"
DATA_DIR    = str(_HERE.parent / "data")  # absolute: .../data/ (gitignored, not shipped)

PING_SEC    = 20    # heartbeat interval (Bybit requires < 30s)
CVD_WINDOW  = 300   # rolling CVD window in seconds (5 min)
LIQ_KEEP    = 50    # liquidations to keep per symbol in Redis
FLUSH_SEC   = 5     # Redis flush interval
RECONNECT_SEC = 5   # delay between reconnect attempts
