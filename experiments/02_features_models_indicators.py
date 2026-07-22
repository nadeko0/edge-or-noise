"""Where is the predictive signal actually coming from?

1. Feature ablation: order-flow-only vs price/volatility-only features.
2. Model engine comparison: does swapping XGBoost for LightGBM /
   CatBoost / RandomForest change anything.
3. A single-variable logistic regression on rolling realized
   volatility alone, compared against the full model -- this is the
   test that shows the "edge" is ~99% explained by one variable.
4. The project's own technical-indicator library (see indicators/ in
   the original research repo) as a feature set.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import FEES, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, FEATURE_COLS_ORDERFLOW, FEATURE_COLS_PRICEVOL
from tickml import label_triple_barrier, walk_forward


def labeled(df):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        PT, SL0, MAX_BARS, FEES, True,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def logistic_walk_forward(df, feature_cols, n_folds=3):
    from sklearn.metrics import roc_auc_score
    d = df[df["label"] >= 0].dropna(subset=feature_cols).reset_index(drop=True)
    n = len(d)
    cuts = [int(n * f) for f in np.linspace(0.55, 1.0, n_folds + 1)]
    aucs = []
    for i in range(n_folds):
        test_start, test_end = cuts[i], cuts[i + 1]
        train = d.iloc[:max(test_start - 10, 0)]
        test = d.iloc[test_start:test_end]
        if len(train) < 500 or len(test) < 100:
            continue
        mu, sd = train[feature_cols].mean(), train[feature_cols].std() + 1e-9
        model = LogisticRegression(max_iter=500)
        model.fit((train[feature_cols] - mu) / sd, train["label"])
        proba = model.predict_proba((test[feature_cols] - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(test["label"], proba))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    btc = labeled(get_live("BTCUSDT"))

    print("== feature ablation ==")
    r_full = walk_forward(btc, FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)
    r_flow = walk_forward(btc, FEATURE_COLS_ORDERFLOW, max_bars=MAX_BARS, verbose=False)
    r_price = walk_forward(btc, FEATURE_COLS_PRICEVOL, max_bars=MAX_BARS, verbose=False)
    print(f"full ({len(FEATURE_COLS_FULL)} features):      AUC={r_full['mean_auc']:.4f}")
    print(f"orderflow-only ({len(FEATURE_COLS_ORDERFLOW)}): AUC={r_flow['mean_auc']:.4f}")
    print(f"price/vol-only ({len(FEATURE_COLS_PRICEVOL)}):  AUC={r_price['mean_auc']:.4f}")

    print("\n== model engines ==")
    for kind in ("xgb", "lgbm", "catboost", "rf"):
        r = walk_forward(btc, FEATURE_COLS_FULL, model_kind=kind, max_bars=MAX_BARS, verbose=False)
        print(f"{kind:10s}: AUC={r['mean_auc']:.4f}")

    print("\n== single-variable check: logistic regression on vol_20 alone ==")
    auc_vol_only = logistic_walk_forward(btc, ["vol_20"])
    auc_logit_full = logistic_walk_forward(btc, FEATURE_COLS_FULL)
    print(f"logistic(vol_20 only):        AUC={auc_vol_only:.4f}")
    print(f"logistic(all {len(FEATURE_COLS_FULL)} features): AUC={auc_logit_full:.4f}")
    print(f"xgboost(all {len(FEATURE_COLS_FULL)} features):  AUC={r_full['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
