"""Runs the full pipeline on synthetic data, so anyone can check that it
works without the real (non-redistributable) dataset.

Two worlds from tickml/synthetic.py, same generator and seeds:
  - clustered: volatility switches between calm and volatile regimes,
    the property the real-data study found predictive
  - null: constant volatility, nothing to find

What a correct pipeline should print: walk-forward AUC clearly above
0.5 in the clustered world, close to 0.5 in the null world. If the null
world came out well above 0.5, the pipeline would be leaking future
information somewhere; if the clustered world did not, it would be
failing to find a signal that is definitely there.

Needs no data and no environment variables:
  uv run experiments/00_synthetic_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tickml import FEATURE_COLS_FULL, add_core_features, label_triple_barrier, walk_forward
from tickml.loaders_live import build_bar_dataset
from tickml.synthetic import write_synthetic_dataset

PT, SL0, MAX_BARS, FEES = 0.0020, 0.0, 5, 0.00040  # same as experiments/common.py
N_DAYS = 6
SEEDS = (0, 1, 2, 3, 4, 5)


def world_auc(vol_clustering: bool, seed: int) -> float:
    with tempfile.TemporaryDirectory() as td:
        dates = write_synthetic_dataset(Path(td), n_days=N_DAYS, vol_clustering=vol_clustering, seed=seed)
        df = add_core_features(build_bar_dataset("BTCUSDT", dates, Path(td)))
    label, pnl = label_triple_barrier(
        df["open"].values.astype("float64"), df["high"].values.astype("float64"),
        df["low"].values.astype("float64"), df["close"].values.astype("float64"),
        PT, SL0, MAX_BARS, FEES, True,
    )
    df["label"], df["trade_pnl"] = label, pnl
    return walk_forward(df, FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)["mean_auc"]


def main():
    print(f"walk-forward AUC on {N_DAYS} synthetic days, max_bars={MAX_BARS}")
    print(f"{'seed':>4}  {'clustered':>9}  {'null':>6}")
    for seed in SEEDS:
        print(f"{seed:>4}  {world_auc(True, seed):9.4f}  {world_auc(False, seed):6.4f}")


if __name__ == "__main__":
    main()
