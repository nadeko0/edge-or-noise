"""Is the AUC itself distinguishable from chance, or could a model this
flexible produce AUC~0.6 on pure noise? Null distribution built by
training the same model on label-shuffled data -- if the real AUC
exceeds every (or nearly every) shuffled run, the ranking ability is
not an artifact of model capacity or the specific train/test split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from common import FEES, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, make_model


def main():
    btc = get_live("BTCUSDT")
    label, pnl = label_triple_barrier(
        btc["open"].values.astype("float64"), btc["high"].values.astype("float64"),
        btc["low"].values.astype("float64"), btc["close"].values.astype("float64"),
        PT, SL0, MAX_BARS, FEES, True,
    )
    btc["label"], btc["trade_pnl"] = label, pnl

    d = btc[btc["label"] >= 0].dropna(subset=FEATURE_COLS_FULL).reset_index(drop=True)
    n = len(d)
    tr_end = int(n * 0.85)
    train, test = d.iloc[:tr_end], d.iloc[tr_end:]

    model = make_model("xgb")
    model.fit(train[FEATURE_COLS_FULL], train["label"],
              eval_set=[(test[FEATURE_COLS_FULL], test["label"])], verbose=False)
    real_auc = roc_auc_score(test["label"], model.predict_proba(test[FEATURE_COLS_FULL])[:, 1])

    rng = np.random.default_rng(42)
    null_aucs = []
    y_tr = train["label"].values.copy()
    for _ in range(15):
        y_shuffled = rng.permutation(y_tr)
        m = make_model("xgb", early_stopping_rounds=None, n_estimators=150)
        m.fit(train[FEATURE_COLS_FULL], y_shuffled)
        p = m.predict_proba(test[FEATURE_COLS_FULL])[:, 1]
        null_aucs.append(roc_auc_score(test["label"], p))
    null_aucs = np.array(null_aucs)
    p_value = float((null_aucs >= real_auc).sum() / len(null_aucs))

    print(f"real AUC:        {real_auc:.4f}")
    print(f"null mean AUC:   {null_aucs.mean():.4f}  (label-shuffled, {len(null_aucs)} runs)")
    print(f"null max AUC:    {null_aucs.max():.4f}")
    print(f"p-value:         {p_value:.4f}  (fraction of null runs >= real AUC)")


if __name__ == "__main__":
    main()
