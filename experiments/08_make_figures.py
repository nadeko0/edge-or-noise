"""Generates the 3 PNG figures embedded in the root README -- each one
answers exactly one question at a glance, no reading required:

  figures/feature_importance.png  -- what actually predicts profitability?
  figures/auc_by_horizon.png      -- how does ranking ability change with horizon?
  figures/permutation_test.png    -- is the AUC real, or could random labels do this?

Run: uv run experiments/08_make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEES, MAX_BARS, PT, SL0, get_live
from tickml import FEATURE_COLS_FULL, label_triple_barrier, make_model, walk_forward

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

# -- palette (see dataviz skill reference) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def labeled(df, max_bars=MAX_BARS):
    d = df.copy()
    label, pnl = label_triple_barrier(
        d["open"].values.astype("float64"), d["high"].values.astype("float64"),
        d["low"].values.astype("float64"), d["close"].values.astype("float64"),
        PT, SL0, max_bars, FEES, True,
    )
    d["label"], d["trade_pnl"] = label, pnl
    return d


def make_feature_importance_figure(btc):
    d = labeled(btc)
    dd = d[d["label"] >= 0].dropna(subset=FEATURE_COLS_FULL).reset_index(drop=True)
    n = len(dd)
    train, test = dd.iloc[:int(n * 0.85)], dd.iloc[int(n * 0.85):]
    model = make_model("xgb")
    model.fit(train[FEATURE_COLS_FULL], train["label"],
              eval_set=[(test[FEATURE_COLS_FULL], test["label"])], verbose=False)

    importances = model.feature_importances_
    order = np.argsort(importances)[-12:]  # top 12, ascending for barh
    names = [FEATURE_COLS_FULL[i] for i in order]
    values = importances[order]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = [ORANGE if v == values.max() else BLUE for v in values]
    ax.barh(names, values, color=colors, height=0.65)
    ax.set_xlabel("XGBoost feature importance (gain)", fontsize=13)
    ax.set_title("What actually predicts short-horizon profitability?",
                 fontsize=18, fontweight="bold", color=INK, pad=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=12)
    fig.text(0.5, 0.015,
             "vol_20 / range_bar (volatility) dominate -- order flow (CVD, OBI, liquidations) barely register",
             ha="center", va="bottom", fontsize=13, color=INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)
    print("wrote figures/feature_importance.png")


def make_auc_by_horizon_figure(btc):
    horizons = [1, 2, 3, 5, 10]
    aucs = []
    for mb in horizons:
        d = labeled(btc, max_bars=mb)
        r = walk_forward(d, FEATURE_COLS_FULL, max_bars=mb, verbose=False)
        aucs.append(r["mean_auc"])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(horizons))
    bars = ax.bar(x, aucs, color=BLUE, width=0.55, zorder=3)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(-0.4, 0.516, "0.5 = no skill (coin flip)",
            ha="left", fontsize=13, color=INK_SECONDARY)
    for xi, v in zip(x, aucs):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=14, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in horizons], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("prediction horizon (max_bars)", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_ylim(0.45, 0.75)
    ax.set_title("Ranking ability shrinks as the prediction horizon grows",
                 fontsize=18, fontweight="bold", color=INK, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auc_by_horizon.png", dpi=160)
    plt.close(fig)
    print("wrote figures/auc_by_horizon.png")


def make_permutation_test_figure(btc, n_perms=60):
    d = labeled(btc)
    dd = d[d["label"] >= 0].dropna(subset=FEATURE_COLS_FULL).reset_index(drop=True)
    n = len(dd)
    tr_end = int(n * 0.85)
    train, test = dd.iloc[:tr_end], dd.iloc[tr_end:]

    model = make_model("xgb")
    model.fit(train[FEATURE_COLS_FULL], train["label"],
              eval_set=[(test[FEATURE_COLS_FULL], test["label"])], verbose=False)
    real_auc = roc_auc_score(test["label"], model.predict_proba(test[FEATURE_COLS_FULL])[:, 1])

    rng = np.random.default_rng(42)
    null_aucs = []
    y_tr = train["label"].values.copy()
    for _ in range(n_perms):
        y_shuffled = rng.permutation(y_tr)
        m = make_model("xgb", early_stopping_rounds=None, n_estimators=150)
        m.fit(train[FEATURE_COLS_FULL], y_shuffled)
        p = m.predict_proba(test[FEATURE_COLS_FULL])[:, 1]
        null_aucs.append(roc_auc_score(test["label"], p))
    null_aucs = np.array(null_aucs)
    p_value = float((null_aucs >= real_auc).sum() / len(null_aucs))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(null_aucs, bins=14, color=BLUE, alpha=0.85, zorder=3,
            label=f"AUC with shuffled labels (n={n_perms})")
    ax.axvline(real_auc, color=ORANGE, linewidth=2.5, zorder=4)
    ymax = max(np.histogram(null_aucs, bins=14)[0]) * 1.15
    ax.set_ylim(0, ymax)
    ax.text(real_auc + 0.004, ymax * 0.92, f"observed AUC = {real_auc:.3f}\np = {p_value:.4f}",
            color=ORANGE, fontsize=15, fontweight="bold", va="top")
    ax.set_xlabel("ROC-AUC", fontsize=13)
    ax.set_ylabel("count", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Real signal vs. random-label noise",
                 fontsize=18, fontweight="bold", color=INK, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=13, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_test.png", dpi=160)
    plt.close(fig)
    print(f"wrote figures/permutation_test.png (p={p_value:.4f})")


def main():
    btc = get_live("BTCUSDT")
    make_feature_importance_figure(btc)
    make_auc_by_horizon_figure(btc)
    make_permutation_test_figure(btc)


if __name__ == "__main__":
    main()
