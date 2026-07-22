"""Validation primitives: chronological walk-forward with a purge gap,
a label-shuffle permutation test, an independent-dataset OOS check, and
a sequential (one-position-at-a-time) equity-curve simulator.

Design choices that matter for correctness:
  - walk_forward uses an EXPANDING training window and drops the last
    `max_bars + 1` training rows before each test window (purge) so a
    training label can never depend on an outcome bar that falls inside
    the held-out test window.
  - oos_eval trains once on ALL of one dataset and scores an entirely
    different dataset (different symbol and/or different time period)
    -- this is the check that catches "found a pattern that only
    exists in the window it was mined on".
  - permutation_test builds a null distribution by circularly
    time-shifting which bars count as "signal", not by shuffling i.i.d.
    -- this preserves the autocorrelation structure of the PnL series
    under the null, which a naive shuffle would destroy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .labeling import profit_factor
from .models import make_model


def walk_forward(df: pd.DataFrame, feature_cols: list[str], model_kind: str = "xgb",
                  n_folds: int = 3, max_bars: int = 5, verbose: bool = True) -> dict:
    d = df[df["label"] >= 0].dropna(subset=feature_cols).reset_index(drop=True)
    n = len(d)
    cuts = [int(n * f) for f in np.linspace(0.55, 1.0, n_folds + 1)]
    fold_aucs, fold_pfs = [], []
    for i in range(n_folds):
        test_start, test_end = cuts[i], cuts[i + 1]
        purge = max_bars + 1
        train = d.iloc[:max(test_start - purge, 0)]
        test = d.iloc[test_start:test_end]
        if len(train) < 500 or len(test) < 100:
            continue
        X_tr, y_tr = train[feature_cols], train["label"]
        X_te, y_te = test[feature_cols], test["label"]
        if model_kind == "xgb":
            model = make_model("xgb")
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        else:
            model = make_model(model_kind)
            model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        base_pf = profit_factor(test["trade_pnl"].values)
        fold_aucs.append(auc)
        fold_pfs.append(base_pf)
        if verbose:
            print(f"    fold{i + 1}: n_test={len(test):,} AUC={auc:.4f} base_PF={base_pf:.3f}")
    return dict(
        mean_auc=float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        fold_aucs=fold_aucs,
        mean_base_pf=float(np.mean(fold_pfs)) if fold_pfs else float("nan"),
        n_folds_run=len(fold_aucs),
    )


def walk_forward_predictions(df: pd.DataFrame, feature_cols: list[str], model_kind: str = "xgb",
                              n_folds: int = 3, max_bars: int = 5) -> pd.DataFrame:
    """Same folding as walk_forward, but returns the concatenated
    out-of-sample rows with a `proba` column -- every bar is scored
    only by a model that never trained on it, so the rows can be
    stitched into one honest equity curve instead of per-fold summaries."""
    d = df[df["label"] >= 0].dropna(subset=feature_cols).reset_index(drop=True)
    n = len(d)
    cuts = [int(n * f) for f in np.linspace(0.55, 1.0, n_folds + 1)]
    out = []
    for i in range(n_folds):
        test_start, test_end = cuts[i], cuts[i + 1]
        purge = max_bars + 1
        train = d.iloc[:max(test_start - purge, 0)]
        test = d.iloc[test_start:test_end].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        X_tr, y_tr = train[feature_cols], train["label"]
        if model_kind == "xgb":
            model = make_model("xgb")
            model.fit(X_tr, y_tr, eval_set=[(test[feature_cols], test["label"])], verbose=False)
        else:
            model = make_model(model_kind)
            model.fit(X_tr, y_tr)
        test["proba"] = model.predict_proba(test[feature_cols])[:, 1]
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def oos_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train once on `train_df` in full, score `test_df` (a different
    symbol and/or a different historical period) that was never used
    for training or threshold selection."""
    train = train_df[train_df["label"] >= 0].dropna(subset=feature_cols)
    test = test_df[test_df["label"] >= 0].dropna(subset=feature_cols)
    model = make_model("xgb")
    n_val = max(int(len(train) * 0.05), 50)
    tr, va = train.iloc[:-n_val], train.iloc[-n_val:]
    model.fit(tr[feature_cols], tr["label"], eval_set=[(va[feature_cols], va["label"])], verbose=False)
    proba = model.predict_proba(test[feature_cols])[:, 1]
    auc = roc_auc_score(test["label"], proba)
    base_pf = profit_factor(test["trade_pnl"].values)
    return dict(auc=float(auc), base_pf=base_pf, n_test=len(test))


def permutation_test(pnl_all: np.ndarray, signal_mask: np.ndarray, n_perms: int = 500, seed: int = 7):
    """Null distribution via circular time-shift of which bars count as
    'signal' -- preserves the PnL series' own autocorrelation, unlike a
    naive i.i.d. shuffle. Returns None if there are too few signals."""
    n = len(pnl_all)
    fire_idx = np.where(signal_mask)[0]
    if len(fire_idx) < 10:
        return None
    obs_pf = profit_factor(pnl_all[signal_mask])
    rng = np.random.default_rng(seed)
    perm_pfs = []
    for _ in range(n_perms):
        shift = int(rng.integers(1, n))
        shifted_idx = (fire_idx + shift) % n
        perm_pfs.append(profit_factor(pnl_all[shifted_idx]))
    perm_pfs = np.array(perm_pfs)
    p_value = float((perm_pfs >= obs_pf).sum() / n_perms)
    return dict(n=len(fire_idx), obs_pf=round(obs_pf, 3), p_value=p_value,
                null_median=round(float(np.median(perm_pfs)), 3))


def equity_curve(pred_df: pd.DataFrame, threshold: float, max_bars: int, bar_ms: int,
                  start_capital: float = 1000.0, position_fraction: float = 1.0) -> dict:
    """Sequential, one-position-at-a-time compounding simulation. Skips
    any signal that fires while a previous trade is still open
    (max_bars minutes after its own entry) -- a single account cannot
    hold overlapping instances of the same fixed-horizon trade."""
    d = pred_df.sort_values("bucket").reset_index(drop=True)
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    n_trades = 0
    in_trade_until = -1
    trade_pnls = []
    curve = []
    for row in d.itertuples():
        if row.bucket < in_trade_until:
            continue
        if row.proba < threshold:
            continue
        pnl_pct = row.trade_pnl
        stake = equity * position_fraction
        equity += stake * pnl_pct
        trade_pnls.append(pnl_pct)
        n_trades += 1
        in_trade_until = row.bucket + max_bars * bar_ms
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        curve.append((row.bucket, equity))
    trade_pnls = np.array(trade_pnls)
    sharpe = (
        (trade_pnls.mean() / (trade_pnls.std() + 1e-12)) * np.sqrt(len(trade_pnls))
        if len(trade_pnls) > 1 else float("nan")
    )
    return dict(
        n_trades=n_trades,
        final_equity=round(equity, 2),
        total_return_pct=round((equity / start_capital - 1) * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        win_rate_pct=round(float((trade_pnls > 0).mean() * 100), 1) if n_trades else None,
        pf=round(profit_factor(trade_pnls), 3) if n_trades else None,
        sharpe_like=round(float(sharpe), 3) if sharpe == sharpe else None,
        curve=curve,
    )
