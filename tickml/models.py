"""Thin factory over 4 model engines, kept interchangeable on purpose --
one of the findings in this research is that model choice does not
matter (see reports/FINAL_REPORT.md), which is only demonstrable if
swapping engines is a one-line change.
"""
from __future__ import annotations


def make_model(kind: str = "xgb", **overrides):
    if kind == "xgb":
        import xgboost as xgb
        params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      n_jobs=2, eval_metric="logloss", early_stopping_rounds=30)
        params.update(overrides)
        return xgb.XGBClassifier(**params)
    elif kind == "lgbm":
        import lightgbm as lgbm
        params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      n_jobs=2, verbosity=-1)
        params.update(overrides)
        return lgbm.LGBMClassifier(**params)
    elif kind == "catboost":
        from catboost import CatBoostClassifier
        params = dict(iterations=300, depth=4, learning_rate=0.05,
                      l2_leaf_reg=3.0, thread_count=2, verbose=False)
        params.update(overrides)
        return CatBoostClassifier(**params)
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        params = dict(n_estimators=300, max_depth=6, min_samples_leaf=20,
                      n_jobs=2, random_state=42)
        params.update(overrides)
        return RandomForestClassifier(**params)
    raise ValueError(f"unknown model kind: {kind}")
