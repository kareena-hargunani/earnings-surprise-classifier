"""
_chow_test.py
-------------
Diagnostic script (not part of the core pipeline): tests whether the FF3
factor loadings (alpha, beta_mkt, beta_smb, beta_hml) are stable across each
event's estimation window (-252 to -10 days), using a fixed-midpoint Chow
test.

For each event:
    1. Fit FF3 on the full estimation window (restricted/pooled model).
    2. Split the window in half; fit FF3 separately on each half
       (unrestricted model).
    3. Chow F-test: compares RSS of the pooled fit against the sum of RSS
       from the two half-fits. A significant result (p < 0.05) means the
       factor loadings differ across the two halves -- i.e., a detected
       structural break.

Reports the overall incidence rate of detected breaks, and the incidence
rate broken out by walk-forward quarter/regime, matching the same
regime buckets used in model.py's subperiod_analysis.

This is a retrospective diagnostic on the label-construction methodology,
not a forward-looking feature -- see README "Known Limitations" #6.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ESTIMATION_WINDOW = (-252, -10)
MIN_ESTIMATION_DAYS = 120
N_PARAMS = 4  # const, mkt_rf, smb, hml

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def assign_regime(q: pd.Period) -> str:
    if q < pd.Period("2020Q1", freq="Q"):
        return "Pre-COVID"
    elif q <= pd.Period("2020Q4", freq="Q"):
        return "COVID"
    elif q <= pd.Period("2021Q4", freq="Q"):
        return "Post-COVID"
    else:
        return "Rate Hike Cycle"


def chow_test_for_event(ticker, earnings_date, prices, ff):
    """Returns (p_value, n1, n2) or None if insufficient data."""
    t_prices = prices[prices["ticker"] == ticker].set_index("date")["adj_close"].sort_index()
    if len(t_prices) < 2:
        return None
    returns = np.log(t_prices / t_prices.shift(1)).dropna()

    all_dates = returns.index
    try:
        event_idx = all_dates.get_loc(earnings_date)
    except KeyError:
        future = all_dates[all_dates >= earnings_date]
        if len(future) == 0:
            return None
        event_idx = all_dates.get_loc(future[0])

    est_start_idx = event_idx + ESTIMATION_WINDOW[0]
    est_end_idx = event_idx + ESTIMATION_WINDOW[1]
    if est_start_idx < 0 or est_end_idx < 0:
        return None

    est_dates = all_dates[est_start_idx: est_end_idx + 1]
    if len(est_dates) < MIN_ESTIMATION_DAYS:
        return None

    stock_ret = returns.loc[est_dates]
    ff_est = ff.set_index("date").loc[ff.set_index("date").index.isin(est_dates)]
    df = pd.DataFrame({"stock_ret": stock_ret}).join(ff_est, how="inner")
    df["excess_ret"] = df["stock_ret"] - df["rf"]
    df = df.dropna()
    if len(df) < MIN_ESTIMATION_DAYS:
        return None

    n = len(df)
    mid = n // 2
    half1, half2 = df.iloc[:mid], df.iloc[mid:]
    if len(half1) < N_PARAMS + 5 or len(half2) < N_PARAMS + 5:
        return None

    def rss(sub):
        X = sm.add_constant(sub[["mkt_rf", "smb", "hml"]])
        y = sub["excess_ret"].values
        try:
            m = sm.OLS(y, X).fit()
        except (ValueError, np.linalg.LinAlgError):
            return None
        if np.any(np.isnan(m.params)):
            return None
        return float(np.sum(m.resid ** 2))

    rss_pooled = rss(df)
    rss_1 = rss(half1)
    rss_2 = rss(half2)
    if rss_pooled is None or rss_1 is None or rss_2 is None:
        return None

    n1, n2 = len(half1), len(half2)
    df_num = N_PARAMS
    df_den = n1 + n2 - 2 * N_PARAMS
    if df_den <= 0:
        return None

    numerator = (rss_pooled - (rss_1 + rss_2)) / df_num
    denominator = (rss_1 + rss_2) / df_den
    if denominator <= 0:
        return None

    f_stat = numerator / denominator
    p_value = sp_stats.f.sf(f_stat, df_num, df_den)
    return p_value, n1, n2


def main():
    print("Loading data...")
    labeled = pd.read_parquet(os.path.join(DATA_DIR, "processed", "labeled_events.parquet"))
    prices = pd.read_parquet(os.path.join(DATA_DIR, "raw", "prices.parquet"))
    ff = pd.read_csv(os.path.join(DATA_DIR, "raw", "fama_french.csv"), parse_dates=["date"])

    print(f"Running Chow test on {len(labeled):,} events...")
    results = []
    for i, row in labeled.iterrows():
        if i % 2000 == 0:
            print(f"  Progress: {i:,}/{len(labeled):,}...")
        out = chow_test_for_event(row["ticker"], row["earnings_date"], prices, ff)
        if out is not None:
            p_value, n1, n2 = out
            results.append({
                "ticker": row["ticker"],
                "earnings_date": row["earnings_date"],
                "p_value": p_value,
                "break_detected": p_value < 0.05,
            })

    res_df = pd.DataFrame(results)
    res_df["quarter"] = pd.to_datetime(res_df["earnings_date"]).dt.to_period("Q")
    res_df["regime"] = res_df["quarter"].apply(assign_regime)

    print(f"\nEvents tested: {len(res_df):,} (of {len(labeled):,} labeled events)")
    overall_rate = res_df["break_detected"].mean()
    print(f"\nOverall structural break incidence (p < 0.05): {overall_rate:.1%}")
    print("(Expected under pure chance with no true breaks anywhere: ~5.0%)")

    print("\nIncidence by regime:")
    by_regime = res_df.groupby("regime").agg(
        n_events=("break_detected", "count"),
        incidence=("break_detected", "mean"),
    ).round(3)
    print(by_regime.to_string())

    print("\nIncidence by quarter (walk-forward fold):")
    by_quarter = res_df.groupby("quarter").agg(
        n_events=("break_detected", "count"),
        incidence=("break_detected", "mean"),
    ).round(3)
    print(by_quarter.to_string())

    out_path = os.path.join(DATA_DIR, "processed", "chow_test_results.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved event-level results to {out_path}")

    by_quarter.to_csv(os.path.join(DATA_DIR, "processed", "chow_test_by_quarter.csv"))
    by_regime.to_csv(os.path.join(DATA_DIR, "processed", "chow_test_by_regime.csv"))


if __name__ == "__main__":
    main()
