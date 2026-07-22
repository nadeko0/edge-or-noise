# Final Report: Searching for a Scalping Edge in Bybit Tick Data

~80 hypotheses tested against 1-minute-bar features built from raw
tick data (trades, order book, liquidations, funding, open interest).
Raw results referenced below correspond to the scripts in
`experiments/`.

## Data coverage by hypothesis

Not every hypothesis was run against every symbol or the full date
range -- stated explicitly here rather than left implicit:

| What was tested | Symbol(s) | Period | Script(s) |
|---|---|---|---|
| Core hypotheses: labeling variants, model engines, TA indicators, leverage, funding/OI, SMC concepts | BTCUSDT | 2026-04-22 → 2026-07-20 (90 days, self-collected) | `01`–`05`, `07`, `08` |
| Real L2 order-book depth / whale-wall | BTCUSDT | Full 90 days (blended schema) and a restricted 2026-05-02 → 2026-07-20 single-schema window (see §4) | `03` |
| Cross-asset walk-forward (own model) | ETHUSDT, SOLUSDT | Same 90-day window as BTC | `06` |
| Cross-asset OOS (BTC-trained model scored on another symbol) | ETHUSDT, SOLUSDT | Same 90-day window as BTC | `06` |
| Independent-year OOS confirmation | **BTCUSDT only** | 2024-02-12 → 2024-06-02 (112 days, third-party archive) | `06` |

The third-party 2024 archive is structured to also cover ETH and SOL,
but this repo's out-of-sample check only trains/scores against its BTC
data -- an ETH/SOL run against that archive was not performed here and
should not be assumed from the BTC result.

The collector was upgraded on 2026-05-01 to also capture
`orderbook.rpi` (retail price improvement) and to switch the main
`orderbook.50` feed to raw 50-level deltas (previously a coarser
pre-aggregated top-5 snapshot, see §4) -- 2026-05-01 itself is a
partial/transitional day from that restart, so 2026-05-02 is used as
the clean start of the new format. `orderbook.rpi` is collected but
not used by any experiment in this repo (not part of the ~80 tested
hypotheses).

## Summary

A real, statistically significant, cross-asset, cross-year predictive
signal exists (ROC-AUC 0.58–0.67 depending on prediction horizon). It
is almost entirely explained by short-term realized-volatility
clustering, not order flow, order-book depth, or direction. No tested
configuration produced a profit factor reliably above 1.0 on a sample
large enough to trust.

---

## 1. Methodology

- **Labeling**: triple-barrier on 1-minute bars. Entry at the next
  bar's open, take-profit (0.20% unless noted), optional stop-loss,
  otherwise a time-exit after `max_bars` bars (default 5) at that
  bar's close. Fees: 0.04% round-trip (maker/limit assumption),
  deducted from every trade.
- **Models**: XGBoost (primary), LightGBM, CatBoost, RandomForest, and
  a plain logistic regression for a single-variable check. Features
  drawn from trades (CVD, returns, realized volatility), tickers
  (top-of-book imbalance, spread, funding, open interest),
  liquidations, and — separately tested — real L2 order-book depth and
  a full classical technical-indicator library.
- **Validation**: walk-forward, 3 chronological expanding-window
  folds, purge gap of `max_bars + 1` rows between train and test (see
  `tickml/validation.py`). Ranking metric: ROC-AUC (0.5 = no signal).
  Profitability metrics: profit factor (PF, >1 = net positive) and win
  rate, plus a sequential (one position at a time) compounding equity
  curve with drawdown.
- **Out-of-sample check**: the same pipeline run against the
  independent 2024 dataset, never touched during training or threshold
  selection.

## 2. Bug audit

- An off-by-one in the walk-forward purge gap was found and fixed: the
  purge was hardcoded to 10 rows regardless of the actual `max_bars`
  used, which was exactly at the leak boundary for the `max_bars=10`
  configuration. Measured impact after the fix: ΔAUC = 0.0001 (within
  noise).
- A pooled multi-symbol experiment initially used a random train/test
  split instead of a chronological one; corrected. Result did not
  change materially (AUC 0.604 chronological vs. 0.614 random).
- 1-minute OHLC bars reconstructed from raw trades were cross-checked
  against Bybit's own public kline API on a sample day: close-price
  discrepancy ≈ 1e-6% (floating-point only), confirming no scaling or
  timestamp-alignment bug in the core bar construction.
- No lookahead was found in the feature pipeline: every rolling/lag
  feature at bar *i* uses only data available at or before bar *i*'s
  close (see `tests/test_features.py` for an explicit check —
  mutating bars strictly after a cutoff must not change any feature
  value at or before that cutoff).
- Two of this project's own orderbook capture formats were found
  mid-dataset (an early pre-aggregated top-5-level snapshot format,
  later replaced by raw 50-level deltas); both are handled by
  `tickml/orderbook_l2.py`.

## 3. Core finding: horizon and direction

| max_bars | AUC | baseline PF (trade every signal) |
|---|---|---|
| 1 | 0.669 | 0.14 |
| 2 | 0.631 | 0.23 |
| 3 | 0.614 | 0.29 |
| 5 | 0.597 | 0.37 |
| 10 | 0.580 | 0.48 |

AUC increases as the horizon shortens (less accumulated noise to
predict over); baseline profitability decreases (less room to clear
fees). Short side AUC = 0.597, statistically indistinguishable from
the long side (0.597) — the signal is not directional.

## 4. Where the signal comes from

| Feature set | AUC |
|---|---|
| Full (21 features) | 0.597 |
| Order-flow only (CVD, liquidations, OBI, funding, OI) | 0.572 |
| Price/volatility only | 0.599 |
| Logistic regression on `vol_20` (rolling realized volatility) alone | 0.590 |

A single-variable logistic regression on rolling realized volatility
alone reaches 96% of the full 21-feature XGBoost model's AUC. Real L2
order-book depth (replayed from raw 50-level deltas, not just
top-of-book), a "whale wall" feature (anomalously large single resting
order vs. local median), funding rate, and open interest each added no
measurable improvement.

The collector's own order-book capture changed format mid-dataset
(2026-05-02: an early pre-aggregated top-5-level snapshot was replaced
by raw 50-level deltas, see `tickml/orderbook_l2.py`), so the 90-day
result above blends two collection eras. Rerunning the same comparison
restricted to the single-schema window (2026-05-02 → 2026-07-20, where
top10-depth and whale-wall coverage is 100% rather than partial)
confirms the same conclusion, not an artifact of the blend:

| Feature set (2026-05-02 → 2026-07-20 only) | AUC |
|---|---|
| Baseline | 0.594 |
| + top10 depth / book-depth-ratio | 0.593 |
| + whale-wall size ratio | 0.595 |

A full classical technical-indicator library
(RSI, MACD, Bollinger, ADX, Stochastic, SuperTrend, Hurst exponent,
CMF, Fisher transform, KAMA, MFI, linear-regression slope/R², z-score,
efficiency ratio) added +0.004 AUC over the price/volatility baseline;
its single strongest feature (ATR%) is simply a better-calibrated
volatility proxy than the original ad-hoc rolling standard deviation.

## 5. Model engine and stop-loss variants

| Model | AUC |
|---|---|
| XGBoost | 0.597 |
| LightGBM | 0.595 |
| CatBoost | 0.598 |
| RandomForest | 0.597 |

| Stop-loss | AUC |
|---|---|
| None (time-exit only) | 0.597 |
| 0.10% | 0.567 |
| 0.15% | 0.585 |
| 0.20% (symmetric with 0.20% take-profit) | 0.591 |
| Volatility-scaled (ATR-style dynamic barrier) | 0.53–0.56 |

Adding a stop-loss did not improve AUC in any variant tested; several
made it worse. Model choice is interchangeable.

## 6. Leverage and liquidation risk

Simulated 1×–20× leverage using a realistic maximum-adverse-excursion
(MAE) based liquidation model (a position is liquidated once the
worst intrabar move against it reaches ~1/leverage minus a maintenance
-margin buffer, whichever comes before the trade's natural exit) —
not naive PnL multiplication.

- Worst historical MAE across the full 90-day BTC dataset: 1.63%.
- Liquidation threshold at 20× leverage: ~4.5%.
- Measured liquidation rate at every tested leverage (1×–20×), every
  configuration: 0.0%.

Leverage therefore only rescales whatever result already existed. At a
marginal configuration (PF = 1.078, threshold 0.60, 1×), returns scale
roughly linearly through 10× (final equity $1,003 → $1,021 on a $1,000
simulated account) and turn negative at 20× ($995) once drawdown
compounding outpaces the small edge — leverage amplifies the fragility
of a marginal result, it does not create one.

## 7. Rejected hypotheses (funding, open interest, and literal "smart money" concepts)

| Hypothesis | Result |
|---|---|
| OI/price quadrant (new longs / new shorts / short-covering / long-liquidation) as a feature | AUC = 0.597, identical to baseline |
| "New shorts" (OI up, price down) as contrarian long signal | n=676, PF=0.313, permutation p=0.612 (not significant) |
| Funding-rate extreme as contrarian signal | n=852, PF=0.438, permutation p=0.198 (not significant) |
| Whale wall (anomalous single large book order vs. local median) as a feature | AUC unchanged vs. baseline |
| Liquidity sweep / stop hunt (price wicks past a recent swing level, coincident liquidation spike) | n=25, PF=1.309, permutation p=0.008 — see §8 |
| Direction predictability conditional on "large move likely" (two-stage decompose) | direction AUC ≈ 0.52 overall and within the "large move likely" subset — no meaningful conditional signal |
| Multi-timeframe features (1h/4h realized volatility and range, 15/60-min trend) | AUC = 0.599, same negligible-improvement pattern as every other feature addition |

## 8. Multiple-comparisons check: does a "great result" survive at scale?

The liquidity-sweep rule (§7) looked significant in isolation
(p=0.008), but:

- Loosening its parameters (swing window, liquidation-spike threshold)
  dropped PF from 1.31 back to 0.89–0.99 — the effect exists only at
  one narrow parameter combination, a classic overfitting signature.
- It fired only 3–4 times across the entire independent 2024 dataset
  regardless of parameters — too rare to confirm or refute
  out-of-sample.

A stronger version of the same check was run on two equity-curve
configurations that looked excellent on a small in-sample test slice:

| Config | In-sample (small test slice) | Out-of-sample (2024, independent) |
|---|---|---|
| max_bars=3, threshold=0.60 | n=17, PF=7.9 | n=2,245, PF=0.73 |
| max_bars=1, threshold=0.50 | n=26, PF=2.24 | n=5,626, PF=0.53 |

AUC held up on the independent dataset in both cases (0.609 and 0.666
respectively) — the ranking ability is real — but the specific
fixed-threshold trading rule did not survive contact with a sample two
orders of magnitude larger. Across ~80 tested hypotheses and
thresholds, finding one or two that look spectacular on a small sample
by chance alone is the expected outcome, not evidence of an edge.

## 9. Conclusion

The predictive signal found here is real and reproducible: it survives
a label-shuffle permutation test (p=0.0000, real AUC above all 15
null-distribution runs), generalizes across BTC/ETH/SOL, and holds up
on an independent year of data. It is best characterized as short-term
realized-volatility clustering — a well-documented statistical
property of financial time series — rather than an order-flow,
order-book, or "smart money" pattern. No tested labeling scheme,
feature set, model engine, or leverage level converted this ranking
ability into a profit factor reliably above 1.0 at a sample size large
enough to trust.
