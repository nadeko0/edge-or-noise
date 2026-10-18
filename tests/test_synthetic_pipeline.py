"""End-to-end pipeline check on synthetic data (see tickml/synthetic.py):
the full loaders -> features -> labels -> walk-forward path must find a
planted volatility-clustering signal, and must NOT find one in a world
with constant volatility. The second assertion is the important one: a
pipeline that leaks future information scores well above 0.5 on noise."""
from pathlib import Path

from tickml import FEATURE_COLS_FULL, add_core_features, label_triple_barrier, walk_forward
from tickml.loaders_live import build_bar_dataset
from tickml.synthetic import write_synthetic_dataset


def _auc(tmp_path: Path, vol_clustering: bool) -> float:
    dates = write_synthetic_dataset(tmp_path, n_days=6, vol_clustering=vol_clustering, seed=3)
    df = add_core_features(build_bar_dataset("BTCUSDT", dates, tmp_path))
    label, pnl = label_triple_barrier(
        df["open"].values.astype("float64"), df["high"].values.astype("float64"),
        df["low"].values.astype("float64"), df["close"].values.astype("float64"),
        0.0020, 0.0, 5, 0.00040, True,
    )
    df["label"], df["trade_pnl"] = label, pnl
    return walk_forward(df, FEATURE_COLS_FULL, max_bars=5, verbose=False)["mean_auc"]


def test_pipeline_finds_planted_volatility_signal(tmp_path):
    assert _auc(tmp_path, vol_clustering=True) >= 0.56


def test_pipeline_finds_nothing_in_constant_volatility_world(tmp_path):
    assert _auc(tmp_path, vol_clustering=False) <= 0.54
