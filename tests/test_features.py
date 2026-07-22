"""Feature engineering sanity checks: correct columns are produced, and
mutating a bar strictly AFTER row i never changes row i's features
(the concrete way to check "no lookahead" on a rolling-feature pipeline)."""
import numpy as np
import pandas as pd

from tickml.features import FEATURE_COLS_FULL, add_core_features


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({
        "bucket": np.arange(n) * 60_000,
        "open": price, "high": price + 0.05, "low": price - 0.05, "close": price,
        "volume": rng.uniform(1, 10, n),
        "buy_vol": rng.uniform(0, 5, n), "sell_vol": rng.uniform(0, 5, n),
        "liq_buy_vol": rng.uniform(0, 1, n), "liq_sell_vol": rng.uniform(0, 1, n),
        "obi": rng.uniform(-1, 1, n), "spread": rng.uniform(0, 0.01, n),
        "oi": 1000 + np.cumsum(rng.normal(0, 1, n)), "funding": rng.normal(0, 1e-5, n),
    })
    df["ts"] = pd.to_datetime(df["bucket"], unit="ms", utc=True)
    return df


def test_all_declared_feature_columns_are_produced():
    df = add_core_features(_make_df())
    for col in FEATURE_COLS_FULL:
        assert col in df.columns, f"missing feature column: {col}"


def test_features_at_row_i_do_not_depend_on_future_rows():
    df = _make_df(n=200)
    feat_a = add_core_features(df.copy())

    df_b = df.copy()
    cutoff = 150
    # mutate every bar strictly after `cutoff` -- if there's no lookahead,
    # every feature value at row <= cutoff must be identical to feat_a.
    df_b.loc[cutoff + 1:, ["open", "high", "low", "close", "volume",
                           "buy_vol", "sell_vol", "liq_buy_vol", "liq_sell_vol",
                           "obi", "spread", "oi", "funding"]] *= 3.0
    feat_b = add_core_features(df_b)

    for col in FEATURE_COLS_FULL:
        a = feat_a[col].iloc[:cutoff + 1]
        b = feat_b[col].iloc[:cutoff + 1]
        pd.testing.assert_series_equal(a, b, check_names=False)
