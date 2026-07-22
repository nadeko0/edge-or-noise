"""Paths and constants shared across the package.

DATA_DIR / SFEREZ_DIR are not shipped with this repository (raw tick
data is tens of GB per symbol and is not redistributable). Point the
environment variables below at your own Bybit tick collector output
(see collector/) and/or a similarly-shaped historical dataset before
running anything in experiments/.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TICKML_DATA_DIR", "./data"))
SFEREZ_DIR = Path(os.environ.get("TICKML_SFEREZ_DIR", "./data/historical/sferez"))
CACHE_DIR = Path(os.environ.get("TICKML_CACHE_DIR", "./.cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BAR_MS = 60_000  # 1-minute bars
