"""
01_run_analysis.py
------------------
Master script that ties together data collection, label construction,
feature engineering, and modeling.

Run this end-to-end to reproduce all results.

Workflow:
    1. Load and cache all data sources
    2. Build event study labels (abnormal returns via FF3)
    3. Engineer features (strictly pre-announcement)
    4. Add nonlinear terms (interactions + polynomials)
    5. Run walk-forward CV — expanding window
    6. Run walk-forward CV — rolling window (comparison)
    7. Subperiod analysis
    8. Generate all charts

For development, set N_TICKERS to a small number (e.g. 30).
For the final analysis, set N_TICKERS = None to use all S&P 500.

Expected runtime:
    N_TICKERS=30:   ~5-10 minutes (first run, no cache)
    N_TICKERS=None: ~60-90 minutes (first run, no cache)
    Subsequent runs with cache: ~5 minutes regardless
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_all_data
from features import (
    build_labels, build_features, add_nonlinear_features, get_feature_cols,
    FEATURE_COLS
)
from model import (
    run_walk_forward, subperiod_analysis,
    plot_pr_curve, plot_feature_importance,
    plot_performance_over_time, plot_correlation_heatmap,
)

# ── Config ───────────────────────────────────────────────────────────────────

N_TICKERS = 50          # Set to None for full S&P 500
START_DATE = "2015-01-01"
CACHE = True            # Set to False to force re-pull all data
ROLLING_WINDOW = 12     # Quarters for rolling window comparison
SAVE_DIR = os.path.join(os.path.dirname(__file__), "report")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Step 1: Load Data ─────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

data = load_all_data(n_tickers=N_TICKERS, start=START_DATE, cache=CACHE)
prices    = data["prices"]
earnings  = data["earnings"]
ff        = data["fama_french"]
macro     = data["macro"]
tickers   = data["tickers"]

print(f"\nData loaded:")
print(f"  Prices:        {len(prices):,} rows, {prices['ticker'].nunique()} tickers")
print(f"  Earnings:      {len(earnings):,} events")
print(f"  FF factors:    {len(ff):,} trading days")
print(f"  Macro:         {len(macro):,} rows")

# ── Step 2: Build Labels ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Building event study labels")
print("=" * 60)
print("""
Methodology note:
    Labels are constructed using the Fama-French 3-factor model,
    which is the same event study framework used in academic finance
    and in professional litigation support work. The FF3 model
    provides a cleaner estimate of 'normal' return than simple
    market-adjusted returns because it accounts for size (SMB) and
    value (HML) factor exposures that drive systematic cross-sectional
    differences in expected returns.

    Label = 1 if CAR[0,+1] > 3% (significant positive surprise)
    Label = 0 otherwise

    The 3% threshold is a judgment call — it represents approximately
    one standard deviation of daily returns for large-cap stocks and
    separates economically meaningful moves from noise. Sensitivity
    analysis with different thresholds (1%, 2%, 5%) is included in
    the robustness checks.
""")

labeled = build_labels(
    earnings=earnings,
    prices=prices,
    ff_factors=ff,
)
labeled.to_parquet(
    os.path.join(os.path.dirname(__file__), "data", "processed", "labeled_events.parquet"),
    index=False
)

# ── Step 3: Build Features ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Engineering features")
print("=" * 60)
print("""
Lookahead bias prevention:
    Every feature is computed using data strictly before the earnings
    announcement date. The cutoff is the last trading day before the
    announcement. This prevents the model from seeing any information
    that would not have been available to an investor at the time of
    the prediction.

    This is the most critical implementation discipline in financial ML.
    A model that accidentally uses any post-announcement information will
    appear to have much better performance than it would in live deployment.
""")

feature_df = build_features(
    labeled_earnings=labeled,
    prices=prices,
    earnings=earnings,
    macro=macro,
    ff_factors=ff,
)

# ── Step 4: Add Nonlinear Features ────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Adding interaction and polynomial features")
print("=" * 60)
print("""
Nonlinear feature design rationale:
    Polynomial terms (degree 2) are added for momentum and volatility
    features where extreme values may predict mean reversion rather than
    continuation — a U-shaped or inverted U-shaped relationship that a
    linear term cannot capture.

    Interaction terms are added for economically motivated feature pairs:
    - Momentum × VIX regime: momentum signal strength varies with market noise
    - EPS history × Volatility: surprise persistence varies with uncertainty
    - Market return × Individual momentum: distinguishes stock-specific moves

    L1 regularization (Lasso) will zero out any of these terms that do
    not contribute marginal predictive value, so including them is safe.
    The model self-selects the relevant nonlinear structure.

    CRITICAL: Interactions and polynomials are added BEFORE standardization.
    Standardizing first and then squaring changes the meaning of polynomial
    terms and would be methodologically incorrect.
""")

feature_df_nl = add_nonlinear_features(feature_df)
feature_df_nl.to_parquet(
    os.path.join(os.path.dirname(__file__), "data", "processed", "feature_matrix.parquet"),
    index=False
)

base_feature_cols = [c for c in FEATURE_COLS if c in feature_df_nl.columns]
all_feature_cols = get_feature_cols(feature_df_nl)

print(f"Base features:     {len(base_feature_cols)}")
print(f"Total features:    {len(all_feature_cols)} (after adding interactions/polynomials)")

# ── Step 5: Correlation Heatmap (Diagnostic) ──────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Feature correlation diagnostic")
print("=" * 60)

plot_correlation_heatmap(
    df=feature_df_nl,
    feature_cols=base_feature_cols,
    save_path=os.path.join(SAVE_DIR, "correlation_heatmap.png"),
)

# ── Step 6: Walk-Forward CV — Expanding Window ────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Walk-forward CV — expanding window (Layer 3: full feature set)")
print("=" * 60)
print("""
Walk-forward validation rationale:
    Standard random train/test splits are invalid for financial time series
    because they leak future information into the training set. Walk-forward
    CV enforces temporal ordering: the model is always trained on past data
    and tested on future data, simulating realistic deployment conditions.

    Expanding window: train on all history up to fold k, test on fold k+1.
    This uses maximum available data in each fold but may be sensitive to
    regime changes in distant historical data.
""")

results_expanding = run_walk_forward(
    df=feature_df_nl,
    feature_cols=all_feature_cols,
    rolling_window=None,
)

# ── Step 7: Walk-Forward CV — Rolling Window ──────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 7: Walk-forward CV — rolling window ({ROLLING_WINDOW} quarters)")
print("=" * 60)
print(f"""
Rolling window rationale:
    A rolling window of {ROLLING_WINDOW} quarters (~3 years) downweights
    older, potentially stale relationships. If rolling outperforms expanding,
    it suggests concept drift — older data contains relationships that no
    longer hold and are diluting the current signal.

    Comparing expanding vs. rolling performance is itself an informative
    diagnostic for non-stationarity.
""")

results_rolling = run_walk_forward(
    df=feature_df_nl,
    feature_cols=all_feature_cols,
    rolling_window=ROLLING_WINDOW,
)

# Compare the two
exp_pr = results_expanding["fold_df"]["pr_auc"].mean()
rol_pr = results_rolling["fold_df"]["pr_auc"].mean()
print(f"\nExpanding window mean PR-AUC: {exp_pr:.3f}")
print(f"Rolling window mean PR-AUC:   {rol_pr:.3f}")
if rol_pr > exp_pr + 0.02:
    print("→ Rolling outperforms: evidence of concept drift in older data.")
elif exp_pr > rol_pr + 0.02:
    print("→ Expanding outperforms: more data is beneficial, limited concept drift.")
else:
    print("→ Similar performance: no strong evidence of concept drift.")

# ── Step 8: Subperiod Analysis ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: Subperiod performance analysis")
print("=" * 60)

subperiod_df = subperiod_analysis(results_expanding)
subperiod_df.to_csv(os.path.join(SAVE_DIR, "subperiod_results.csv"))

# ── Step 9: Charts ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9: Generating charts")
print("=" * 60)

plot_pr_curve(
    results_expanding,
    title="Earnings Surprise Classifier — Precision-Recall Curve\n(Pooled walk-forward out-of-sample predictions)",
    save_path=os.path.join(SAVE_DIR, "pr_curve.png"),
)

plot_feature_importance(
    results_expanding,
    top_n=15,
    save_path=os.path.join(SAVE_DIR, "feature_importance.png"),
)

plot_performance_over_time(
    results_expanding,
    save_path=os.path.join(SAVE_DIR, "performance_over_time.png"),
)

# ── Step 10: Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — Summary")
print("=" * 60)

fold_df = results_expanding["fold_df"]
preds = results_expanding["predictions_df"]
baseline = preds["label"].mean()

print(f"""
Results summary:
  Total labeled events:        {len(feature_df_nl):,}
  Walk-forward test folds:     {len(fold_df)}
  Positive class rate:         {baseline:.1%}

  Expanding window:
    Mean PR-AUC:   {fold_df['pr_auc'].mean():.3f} ± {fold_df['pr_auc'].std():.3f}
    Mean ROC-AUC:  {fold_df['roc_auc'].mean():.3f} ± {fold_df['roc_auc'].std():.3f}
    No-skill PR:   {baseline:.3f}
    Lift over baseline: {(fold_df['pr_auc'].mean() / baseline - 1):.1%}

Key assumptions and limitations:
  1. Survivorship bias: S&P 500 current constituents only
  2. Cross-sectional dependence: same-date events share macro shocks
     → Standard errors are understated; effective N < row count
  3. Concept drift: earnings momentum anomalies weaken as they are exploited
  4. Unknown future shocks: fixed effects cannot account for
     unprecedented macro events in the prediction window
  5. Data quality: yfinance EPS estimates are imperfect; WRDS/IBES
     would provide significantly cleaner labels

Output files saved to report/:
  correlation_heatmap.png
  pr_curve.png
  feature_importance.png
  performance_over_time.png
  subperiod_results.csv
""")
