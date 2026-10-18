"""Forward test: the original 90-day study (2026-04-22 -> 2026-07-20)
frozen exactly as published, scored ONCE on the 89 days the collector
gathered afterwards (2026-07-21 -> 2026-10-17). No parameter, feature,
model or threshold below was chosen by looking at the forward window.

Criteria, fixed before the first run:
  C1  ranking signal persists: BTC forward AUC at max_bars=5 >= 0.55
      and above every label-shuffled null run. Refuted if < 0.53.
  C2  horizon ordering persists: AUC decreases from max_bars=1 to 10.
  C3  still volatility: logistic(vol_20) forward AUC >= 90% of the
      full XGBoost forward AUC.
  C4  still no tradeable rule: no (max_bars, threshold) cell reaches
      PF > 1.0 with permutation p <= 0.05 / n_cells (Bonferroni) on
      >= 200 forward trades.
  C5  cross-asset: the BTC-trained model scored on ETH/SOL forward
      keeps AUC >= 0.55.
  C6  order-book depth (corrected snapshot reader, see
      tickml/orderbook_l2.py): adding depth or whale-wall features to
      the frozen model raises forward AUC by >= 0.005. Otherwise the
      original "depth adds nothing" conclusion stands, now on valid
      features. Train window for C6 is 2026-05-02 -> 2026-07-20, the
      single-schema stretch.

Day-quality rule (also fixed in advance): a symbol-day with fewer than
1380 one-minute trade bars (< ~96% of 1440) is dropped from training
and scoring. Features are still computed on the continuous series, so
dropping a day never shifts a rolling window.

Early stopping uses the last 5% of the TRAINING window (as oos_eval
does), never the forward window. The published walk_forward() uses
the test fold as its early-stopping eval_set; section 0 measures how
much that mattered on the original 90 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import FEES, LIVE_END, LIVE_START, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, make_model, walk_forward
from tickml.config import BAR_MS, CACHE_DIR, DATA_DIR
from tickml.labeling import profit_factor
from tickml.orderbook_l2 import build_l2_bars
from tickml.validation import equity_curve, permutation_test

FWD_START, FWD_END = "2026-07-21", "2026-10-17"
MIN_BARS_PER_DAY = 1380
HORIZONS = (1, 2, 3, 5, 10)
RULE_CELLS = [(mb, thr) for mb in (1, 3, 5) for thr in (0.50, 0.55, 0.60)]
N_NULL = 15
L2_START = "2026-05-02"
DEPTH_COLS = ["deep_obi5", "deep_obi10", "book_depth_ratio"]
WHALE_COLS = ["whale_bid_ratio", "whale_ask_ratio"]


def labeled(df, max_bars=MAX_BARS):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        PT, SL0, max_bars, FEES, True,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def load(symbol):
    """One continuous series over both windows, so the forward window's
    first rolling features are not NaN-padded at the split."""
    df = get_live(symbol, LIVE_START, FWD_END)
    day = df["ts"].dt.strftime("%Y-%m-%d")
    counts = day.value_counts()
    bad = sorted(counts[counts < MIN_BARS_PER_DAY].index)
    expected = {d.strftime("%Y-%m-%d") for d in pd.date_range(LIVE_START, FWD_END)}
    missing = sorted(expected - set(counts.index))
    df["day"] = day
    df["good_day"] = ~day.isin(bad)
    print(f"{symbol}: {len(df):,} bars, short days (<{MIN_BARS_PER_DAY} bars) dropped: "
          f"{[(d, int(counts[d])) for d in bad]}, missing days: {missing}")
    return df


def split(d, max_bars=MAX_BARS):
    d = d[(d["label"] >= 0) & d["good_day"]].dropna(subset=FEATURE_COLS_FULL)
    train = d[d["day"] <= LIVE_END]
    fwd = d[d["day"] >= FWD_START]
    # purge: drop training rows whose label window reaches into the forward window
    return train.iloc[:-(max_bars + 1)], fwd


def fit_frozen(train, feature_cols=FEATURE_COLS_FULL, **overrides):
    model = make_model("xgb", **overrides)
    n_val = max(int(len(train) * 0.05), 50)
    tr, va = train.iloc[:-n_val], train.iloc[-n_val:]
    model.fit(tr[feature_cols], tr["label"], eval_set=[(va[feature_cols], va["label"])], verbose=False)
    return model


def logistic_auc(train, test, cols):
    mu, sd = train[cols].mean(), train[cols].std() + 1e-9
    m = LogisticRegression(max_iter=500).fit((train[cols] - mu) / sd, train["label"])
    return roc_auc_score(test["label"], m.predict_proba((test[cols] - mu) / sd)[:, 1])


def early_stopping_audit(btc):
    """Published walk_forward early-stops on the test fold; redo the same
    3 folds on the original 90 days with a train-tail eval set instead."""
    d = labeled(btc)
    d = d[(d["day"] <= LIVE_END) & (d["label"] >= 0)].dropna(subset=FEATURE_COLS_FULL).reset_index(drop=True)
    pub = walk_forward(d, FEATURE_COLS_FULL, max_bars=MAX_BARS, verbose=False)
    n = len(d)
    cuts = [int(n * f) for f in np.linspace(0.55, 1.0, 4)]
    aucs = []
    for i in range(3):
        train = d.iloc[:cuts[i] - (MAX_BARS + 1)]
        test = d.iloc[cuts[i]:cuts[i + 1]]
        m = fit_frozen(train)
        aucs.append(roc_auc_score(test["label"], m.predict_proba(test[FEATURE_COLS_FULL])[:, 1]))
    print(f"published (early stop on test fold): AUC={pub['mean_auc']:.4f}")
    print(f"clean (early stop on train tail):    AUC={np.mean(aucs):.4f}  folds={np.round(aucs, 4).tolist()}")


def main():
    btc = load("BTCUSDT")

    print("\n== 0. early-stopping audit (original 90 days only) ==")
    early_stopping_audit(btc)

    print("\n== C1/C2. frozen BTC model, forward AUC by horizon ==")
    fwd_auc = {}
    for mb in HORIZONS:
        train, fwd = split(labeled(btc, mb), mb)
        m = fit_frozen(train)
        fwd_auc[mb] = roc_auc_score(fwd["label"], m.predict_proba(fwd[FEATURE_COLS_FULL])[:, 1])
        print(f"MaxBars={mb:2d}: train n={len(train):,}  fwd n={len(fwd):,}  fwd AUC={fwd_auc[mb]:.4f}  "
              f"fwd base_PF={profit_factor(fwd['trade_pnl'].values):.3f}")

    train, fwd = split(labeled(btc))
    rng = np.random.default_rng(42)
    null = []
    for _ in range(N_NULL):
        m = make_model("xgb", early_stopping_rounds=None, n_estimators=150)
        m.fit(train[FEATURE_COLS_FULL], rng.permutation(train["label"].values))
        null.append(roc_auc_score(fwd["label"], m.predict_proba(fwd[FEATURE_COLS_FULL])[:, 1]))
    null = np.array(null)
    print(f"null (label-shuffled, {N_NULL} runs): mean={null.mean():.4f} max={null.max():.4f}  "
          f"p={(null >= fwd_auc[MAX_BARS]).mean():.4f}")

    print("\n== C3. is it still volatility? (MaxBars=5) ==")
    vol_auc = logistic_auc(train, fwd, ["vol_20"])
    print(f"logistic(vol_20):  fwd AUC={vol_auc:.4f}  ({vol_auc / fwd_auc[MAX_BARS] * 100:.1f}% of XGBoost)")
    print(f"logistic(all 21):  fwd AUC={logistic_auc(train, fwd, FEATURE_COLS_FULL):.4f}")

    print(f"\n== C4. threshold rules on the forward window ({len(RULE_CELLS)} cells, "
          f"Bonferroni alpha={0.05 / len(RULE_CELLS):.4f}) ==")
    for mb, thr in RULE_CELLS:
        train, fwd = split(labeled(btc, mb), mb)
        m = fit_frozen(train)
        proba = m.predict_proba(fwd[FEATURE_COLS_FULL])[:, 1]
        mask = proba >= thr
        pt = permutation_test(fwd["trade_pnl"].values, mask, n_perms=1000)
        eq = equity_curve(fwd.assign(proba=proba), thr, mb, BAR_MS)
        if pt is None:
            print(f"MaxBars={mb} thr={thr:.2f}: n={int(mask.sum())} (too few)")
            continue
        print(f"MaxBars={mb} thr={thr:.2f}: n={pt['n']:,}  PF={pt['obs_pf']:.3f}  p={pt['p_value']:.4f}  "
              f"null_median={pt['null_median']:.3f}  | sequential: trades={eq['n_trades']:,} "
              f"PF={eq['pf']} ret={eq['total_return_pct']}% maxDD={eq['max_drawdown_pct']}%")

    print("\n== C5. BTC-trained model scored on ETH / SOL forward ==")
    train, _ = split(labeled(btc))
    m = fit_frozen(train)
    for sym in ("ETHUSDT", "SOLUSDT"):
        _, fwd_o = split(labeled(load(sym)))
        auc = roc_auc_score(fwd_o["label"], m.predict_proba(fwd_o[FEATURE_COLS_FULL])[:, 1])
        print(f"{sym}: fwd n={len(fwd_o):,}  AUC={auc:.4f}  base_PF={profit_factor(fwd_o['trade_pnl'].values):.3f}")

    print("\n== C6. order-book depth, corrected reader (BTC) ==")
    cache = CACHE_DIR / f"l2book_snap50_BTCUSDT_{L2_START}_{FWD_END}.parquet"
    if cache.exists():
        l2 = pd.read_parquet(cache)
    else:
        dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(L2_START, FWD_END)]
        l2 = build_l2_bars(DATA_DIR, "BTCUSDT", dates)
        l2.to_parquet(cache)
    merged = btc.merge(l2[["bucket"] + DEPTH_COLS + WHALE_COLS], on="bucket", how="left")
    for col in DEPTH_COLS + WHALE_COLS:
        merged[col] = merged[col].ffill()
    merged = labeled(merged[merged["day"] >= L2_START].reset_index(drop=True))
    for tag, extra in [("baseline", []), ("+ depth", DEPTH_COLS), ("+ whale wall", WHALE_COLS)]:
        cols = FEATURE_COLS_FULL + extra
        d = merged[(merged["label"] >= 0) & merged["good_day"]].dropna(subset=cols)
        train = d[d["day"] <= LIVE_END].iloc[:-(MAX_BARS + 1)]
        fwd = d[d["day"] >= FWD_START]
        m = fit_frozen(train, cols)
        auc = roc_auc_score(fwd["label"], m.predict_proba(fwd[cols])[:, 1])
        print(f"{tag:13s}: train n={len(train):,}  fwd n={len(fwd):,}  fwd AUC={auc:.4f}")


if __name__ == "__main__":
    main()
