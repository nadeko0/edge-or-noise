"""The single most important check in this repository: train once on
the full live dataset, then score an entirely independent dataset --
different year (2024 vs 2026), same collection method never used
during training or threshold selection.

Also demonstrates why this check matters: two probability thresholds
that looked spectacular on a small in-sample test slice (n=17-26
trades, PF 2-8x) are re-tested here against thousands of independent
trades and collapse back toward/below breakeven -- the textbook
signature of a multiple-comparisons false positive, not a real edge.
"""
from __future__ import annotations

from common import FEES, MAX_BARS, PT, SL0, get_live, get_sferez
from tickml import FEATURE_COLS_FULL, label_triple_barrier, oos_eval


def labeled(df, max_bars=MAX_BARS):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        PT, SL0, max_bars, FEES, True,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def main():
    btc_live = labeled(get_live("BTCUSDT"))
    btc_sferez = labeled(get_sferez("BTC"))

    print("== BTC live-trained model scored on BTC 2024 (sferez) ==")
    r = oos_eval(btc_live, btc_sferez, FEATURE_COLS_FULL)
    print(f"OOS AUC={r['auc']:.4f}  baseline_PF={r['base_pf']:.3f}  n_test={r['n_test']:,}")

    print("\n== cross-asset: BTC-trained model scored on ETH / SOL (2026, same period) ==")
    for sym3, sym_full in [("ETH", "ETHUSDT"), ("SOL", "SOLUSDT")]:
        other = labeled(get_live(sym_full))
        r = oos_eval(btc_live, other, FEATURE_COLS_FULL)
        print(f"{sym_full}: AUC={r['auc']:.4f}  baseline_PF={r['base_pf']:.3f}  n_test={r['n_test']:,}")

    print("\n== does a small-sample 'great result' survive at scale? ==")
    # These two exact configs looked excellent on a ~40-day in-sample
    # test slice (see reports/FINAL_REPORT.md, section 4). Re-scored
    # here against sferez's much larger, truly independent sample.
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from tickml.labeling import profit_factor

    for mb, thr in [(3, 0.60), (1, 0.50)]:
        train_df = labeled(get_live("BTCUSDT"), max_bars=mb)
        test_df = labeled(get_sferez("BTC"), max_bars=mb)
        train = train_df[train_df["label"] >= 0].dropna(subset=FEATURE_COLS_FULL)
        test = test_df[test_df["label"] >= 0].dropna(subset=FEATURE_COLS_FULL)
        model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                   n_jobs=2, eval_metric="logloss", early_stopping_rounds=30)
        n_val = max(int(len(train) * 0.05), 50)
        tr, va = train.iloc[:-n_val], train.iloc[-n_val:]
        model.fit(tr[FEATURE_COLS_FULL], tr["label"], eval_set=[(va[FEATURE_COLS_FULL], va["label"])], verbose=False)
        proba = model.predict_proba(test[FEATURE_COLS_FULL])[:, 1]
        mask = proba >= thr
        n_sel = int(mask.sum())
        auc = roc_auc_score(test["label"], proba)
        if n_sel >= 5:
            pnl = test["trade_pnl"].values[mask]
            pf = profit_factor(pnl)
            wr = float((pnl > 0).mean() * 100)
            print(f"MaxBars={mb} thr={thr}: OOS AUC={auc:.4f}  n_selected={n_sel:,}  PF={pf:.3f}  WR={wr:.1f}%")
        else:
            print(f"MaxBars={mb} thr={thr}: OOS AUC={auc:.4f}  n_selected={n_sel} (too few)")


if __name__ == "__main__":
    main()
