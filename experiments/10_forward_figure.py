"""figures/forward_test.png: AUC by horizon in the original study
(walk-forward on 2026-04-22 -> 2026-07-20, as published) next to the
frozen model scored on the 89 later days (2026-07-21 -> 2026-10-17,
experiments/09_forward_test.py). Answers one question at a glance:
did the signal survive data it never saw?

Run: uv run experiments/10_forward_figure.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from common import LIVE_END
from tickml import FEATURE_COLS_FULL, walk_forward


def _load(name):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fwd = _load("09_forward_test")
figs = _load("08_make_figures")  # palette + rcParams
plt = figs.plt


def main():
    btc = fwd.load("BTCUSDT")
    original, forward = [], []
    for mb in fwd.HORIZONS:
        d = fwd.labeled(btc, mb)
        orig = d[(d["day"] <= LIVE_END) & d["good_day"]].reset_index(drop=True)
        original.append(walk_forward(orig, FEATURE_COLS_FULL, max_bars=mb, verbose=False)["mean_auc"])
        train, test = fwd.split(d, mb)
        m = fwd.fit_frozen(train)
        forward.append(roc_auc_score(test["label"], m.predict_proba(test[FEATURE_COLS_FULL])[:, 1]))
        print(f"MaxBars={mb:2d}: original walk-forward={original[-1]:.4f}  forward={forward[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(fwd.HORIZONS))
    w = 0.38
    ax.bar(x - w / 2, original, w, color=figs.BASELINE, zorder=3,
           label="original study, walk-forward (Apr 22 – Jul 20)")
    ax.bar(x + w / 2, forward, w, color=figs.BLUE, zorder=3,
           label="frozen model on 89 unseen days (Jul 21 – Oct 17)")
    for xi, v in zip(x - w / 2, original):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=12, color=figs.INK_SECONDARY)
    for xi, v in zip(x + w / 2, forward):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=12, color=figs.INK, fontweight="bold")
    ax.axhline(0.5, color=figs.INK_MUTED, linewidth=1.2, linestyle="--", zorder=2,
               label="0.5 = no skill (coin flip)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in fwd.HORIZONS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("prediction horizon (max_bars)", fontsize=13)
    ax.set_ylabel("ROC-AUC (BTCUSDT)", fontsize=13)
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Ranking signal held on 89 days the model never saw",
                 fontsize=18, fontweight="bold", color=figs.INK, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(figs.BASELINE)
    ax.grid(axis="y", color=figs.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=12, labelcolor=figs.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(figs.FIG_DIR / "forward_test.png", dpi=160)
    plt.close(fig)
    print("wrote figures/forward_test.png")


if __name__ == "__main__":
    main()
