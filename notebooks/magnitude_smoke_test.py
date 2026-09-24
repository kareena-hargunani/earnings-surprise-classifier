"""
magnitude_smoke_test.py
------------------------
Standalone smoke test (not part of the core pipeline, no imports from or
edits to src/): does pre-announcement information predict the MAGNITUDE of
the earnings-announcement abnormal return, not just whether it crosses the
3% threshold?

Motivation: the classifier's binary label collapses a CAR of 3.1% and a CAR
of 40% into the same outcome. A continuous-DV regression on CAR itself asks
a different, complementary question: conditional on the surprise direction,
how do covariates shift the SIZE of the reaction?

CAR is already a residual (abnormal return = actual return minus the FF3
market-model-expected return, summed over the event window) — no additional
residualizing step is needed. This script just uses the already-computed
`car` column as a continuous DV in place of the thresholded `label`.

Data: reads data/processed/feature_matrix.parquet (covariates) and
data/processed/labeled_events.parquet (current-quarter eps_surprise_pct,
needed for the surprise-sign dummy — NOT the same as feature_matrix's
eps_surprise_last_q, which is the PRIOR quarter's surprise used as a
pre-announcement predictor in the classifier). Both files already exist on
disk from the existing pipeline; nothing needs to be saved out additively.

Model: OLS of car on raw covariates + a positive-surprise dummy +
interactions between that dummy and a handful of covariates already known
to matter from the classifier (realized_vol_30d, vix_relative,
eps_surprise_mean_4q, ret_5d, mkt_ret_10d), with HC1 heteroskedasticity-
robust standard errors.

This is a smoke test: if nothing here is significant / signs are wrong,
the right conclusion is "no evidence of exploitable magnitude structure,"
and this file (only this file) should be deleted.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

RAW_COVARIATES = [
    "eps_surprise_last_q", "eps_beat_last_q", "eps_surprise_mean_4q",
    "eps_surprise_std_4q", "eps_beat_streak", "eps_surprise_trend",
    "ret_5d", "ret_60d", "ret_252d_ex1m",
    "realized_vol_30d", "realized_vol_60d", "vol_ratio",
    "pe_ratio", "vix_level", "treasury_10y", "yield_spread",
    "vix_relative", "mkt_ret_10d",
]

# Covariates interacted with the surprise-sign dummy — chosen because
# they're the features the Lasso classifier found most consistently
# important, and are the ones with a plausible reason to have a direction-
# dependent effect on reaction magnitude (volatility/uncertainty amplifying
# the reaction, momentum/macro context shaping how surprising a beat/miss
# actually is).
INTERACT_WITH = [
    "realized_vol_30d", "vix_relative", "eps_surprise_mean_4q",
    "ret_5d", "mkt_ret_10d",
]


def load_data() -> pd.DataFrame:
    fm = pd.read_parquet(os.path.join(DATA_DIR, "feature_matrix.parquet"))
    le = pd.read_parquet(os.path.join(DATA_DIR, "labeled_events.parquet"))

    # Both files contain a handful of exact-duplicate (ticker, earnings_date)
    # rows (yfinance occasionally returns the same earnings event twice
    # upstream). An inner merge on duplicated keys on both sides fans out
    # (e.g. a key appearing 4x in each produces 16 merged rows), silently
    # multiplying those events' weight in the regression. Drop to one row
    # per event on each side before merging — this is a defensive step
    # local to this script, not a fix applied to the pipeline's data files.
    fm = fm.drop_duplicates(subset=["ticker", "earnings_date"])
    le = le.drop_duplicates(subset=["ticker", "earnings_date"])

    le = le[["ticker", "earnings_date", "eps_surprise_pct", "car"]]
    df = fm.drop(columns=["car", "label"]).merge(
        le, on=["ticker", "earnings_date"], how="inner"
    )
    assert len(df) == len(fm), (
        f"Merge changed row count ({len(fm)} -> {len(df)}); "
        "labeled_events and feature_matrix keys are no longer aligned."
    )
    return df


def winsorize(df: pd.DataFrame, cols: list[str], lower=0.01, upper=0.99) -> pd.DataFrame:
    """
    Clip each column to its [lower, upper] quantile range.

    eps_surprise_pct-style ratios divide by the EPS estimate, which blows
    up when the estimate is near zero — pe_ratio has one event at 2.4e17
    (vs. a median around 19) purely from this. Left unclipped, a single
    row like that dominates the design matrix's condition number and
    silently collapses the HC1 covariance to near-rank-1 (every other
    coefficient's z-stat becomes identical — a numerical artifact, not a
    finding). Winsorizing is a standard, transparent way to keep OLS
    numerically stable without dropping events outright.
    """
    df = df.copy()
    n_clipped_total = 0
    for col in cols:
        lo, hi = df[col].quantile([lower, upper])
        n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
        n_clipped_total += n_clipped
        df[col] = df[col].clip(lo, hi)
    print(f"Winsorized {len(cols)} covariates at [{lower:.0%}, {upper:.0%}] "
          f"quantiles ({n_clipped_total} values clipped in total).")
    return df


def build_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = winsorize(df, RAW_COVARIATES)
    df["pos_surprise"] = (df["eps_surprise_pct"] > 0).astype(int)

    for col in INTERACT_WITH:
        df[f"pos_surprise_x_{col}"] = df["pos_surprise"] * df[col]

    feature_cols = (
        RAW_COVARIATES + ["pos_surprise"]
        + [f"pos_surprise_x_{col}" for col in INTERACT_WITH]
    )

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ["car"])
    n_after = len(df)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after:,} of {n_before:,} events with "
              f"missing values in the covariates used here "
              f"({n_after:,} remain).")

    X = sm.add_constant(df[feature_cols])
    y = df["car"]
    return X, y


def main():
    df = load_data()
    X, y = build_design_matrix(df)

    model = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"\nMagnitude smoke test: CAR ~ covariates + surprise-sign dummy "
          f"+ interactions")
    print(f"n = {int(model.nobs):,}, R² = {model.rsquared:.4f}, "
          f"adj. R² = {model.rsquared_adj:.4f}")
    print("(HC1 heteroskedasticity-robust standard errors)\n")
    print(model.summary())

    print("\n--- Key coefficients ---")
    for name in ["pos_surprise"] + [f"pos_surprise_x_{c}" for c in INTERACT_WITH]:
        coef = model.params[name]
        pval = model.pvalues[name]
        flag = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {name:40s} coef={coef: .5f}  p={pval:.4f}  {flag}")

    n_sig_05 = (model.pvalues.drop("const") < 0.05).sum()
    n_total = len(model.pvalues) - 1
    print(f"\n{n_sig_05}/{n_total} non-intercept coefficients significant at "
          f"p < 0.05.")
    print("Smoke test verdict: inspect the coefficients above for "
          "significance AND plausible signs before deciding whether this "
          "is worth building out further. If nothing here holds up, "
          "delete this file — nothing else in the repo depends on it.")


if __name__ == "__main__":
    main()
