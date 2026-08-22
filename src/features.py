"""
features.py
-----------
Label construction and feature engineering for the Earnings Surprise Classifier.

TWO STAGES:

Stage 1 — Label construction (build_labels)
    For each earnings announcement, compute the abnormal return over
    the [0, +1] day event window using the Fama-French 3-factor model.

Stage 2 — Feature engineering (build_features)
    For each earnings announcement, construct a vector with features derived from
    information that was publicly available prior to the earnings announcement. 
    Lookahead bias is enforced by an explicit date cutoff at every step.

All features are computed as of (earnings_date - 1 trading day).
The StandardScaler is fit within each training fold in model.py
to prevent data leakage from future distributional information. 
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

# Event window: day of announcement [0] and day after [+1] to be inclusive of announcements after market close
EVENT_WINDOW = (0, 1)

# Estimation window for FF model: 252 to 10 trading days before announcement
# We exclude the 10 days immediately before to avoid contamination from
# pre-announcement drift leaking into the normal return estimate
ESTIMATION_WINDOW = (-252, -10)

# Label threshold: abnormal return > this value → positive class (label=1)
# 3% represents roughly 1 standard deviation of daily returns for large caps
LABEL_THRESHOLD = 0.03

# Minimum trading days required in estimation window to compute a valid label
MIN_ESTIMATION_DAYS = 120


# ── Stage 1: Defining the Target: Abnormal Return Labels ─────────────────────

def compute_abnormal_return(
    ticker: str,
    earnings_date: pd.Timestamp,
    prices: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> float | None:
    """
    Compute the cumulative abnormal return (CAR) for a single earnings event.
    
    Method:
        1. Estimate the Fama-French 3-factor model on the estimation window
           (days -252 to -10 relative to earnings_date)
        2. Use estimated coefficients to compute expected returns on
           the event window (days 0 to +1)
        3. CAR = sum of (actual return - expected return) over event window
    
    Returns None if insufficient data is available.
    """
    # Get price series
    ticker_prices = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    ticker_prices = ticker_prices.sort_index()
    
    # Compute daily log returns
    ticker_returns = np.log(ticker_prices / ticker_prices.shift(1)).dropna()
    
    # Get all trading days relative to earnings_date
    all_dates = ticker_returns.index
    try:
        event_idx = all_dates.get_loc(earnings_date)
    except KeyError:
        # Edge case: If earnings date is not a trading day, then find the next trading day
        future_dates = all_dates[all_dates >= earnings_date]
        if len(future_dates) == 0:
            return None
        event_idx = all_dates.get_loc(future_dates[0])
    
    # Define estimation window indices
    est_start_idx = event_idx + ESTIMATION_WINDOW[0]
    est_end_idx = event_idx + ESTIMATION_WINDOW[1]
    
    if est_start_idx < 0 or est_end_idx < 0:
        return None  # If there isn't enough history, return None
    
    est_dates = all_dates[est_start_idx : est_end_idx + 1]
    
    if len(est_dates) < MIN_ESTIMATION_DAYS:
        return None
    
    # Define event window indices
    ev_start_idx = event_idx + EVENT_WINDOW[0]
    ev_end_idx = min(event_idx + EVENT_WINDOW[1], len(all_dates) - 1)
    ev_dates = all_dates[ev_start_idx : ev_end_idx + 1]
    
    if len(ev_dates) == 0:
        return None
    
    # Merge ticker returns with FF factors on the estimation window
    ff = ff_factors.set_index("date")
    
    est_stock = ticker_returns.loc[est_dates]
    est_ff = ff.loc[ff.index.isin(est_dates)]
    
    est_df = pd.DataFrame({"stock_ret": est_stock}).join(est_ff, how="inner")
    est_df["excess_ret"] = est_df["stock_ret"] - est_df["rf"]
    est_df = est_df.dropna()
    
    if len(est_df) < MIN_ESTIMATION_DAYS:
        return None
    
    # FF3 model: excess_ret ~ mkt_rf + smb + hml
    X = sm.add_constant(est_df[["mkt_rf", "smb", "hml"]]) 
    y = est_df["excess_ret"].values
    
    try:
    # X already includes the intercept column (constant), which statsmodels expects.
    # If X does NOT already have it, add it: X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        alpha     = model.params["const"]
        beta_mkt  = model.params["mkt_rf"]
        beta_smb  = model.params["smb"]
        beta_hml  = model.params["hml"]

    # Guard against silent NaN coefficients (degenerate/NaN inputs don't always raise)
        if np.any(np.isnan(model.params)):
            return None
    
    except (ValueError, np.linalg.LinAlgError):
        return None
    
    # Compute expected and abnormal returns on event window
    ev_stock = ticker_returns.loc[ev_dates]
    ev_ff = ff.loc[ff.index.isin(ev_dates)]
    
    ev_df = pd.DataFrame({"stock_ret": ev_stock}).join(ev_ff, how="inner")
    ev_df = ev_df.dropna()
    
    if len(ev_df) == 0:
        return None
    
    expected_ret = (
        alpha
        + beta_mkt * ev_df["mkt_rf"]
        + beta_smb * ev_df["smb"]
        + beta_hml * ev_df["hml"]
        + ev_df["rf"]
    )
    
    abnormal_ret = ev_df["stock_ret"] - expected_ret
    car = abnormal_ret.sum()  # cumulative abnormal return over event window
    
    return car


def build_labels(
    earnings: pd.DataFrame,
    prices: pd.DataFrame,
    ff_factors: pd.DataFrame,
    threshold: float = LABEL_THRESHOLD,
) -> pd.DataFrame:
    """
    Build binary labels for all earnings events.
    
    Label = 1 if CAR > threshold (significant positive surprise)
    Label = 0 otherwise
    
    Returns the earnings DataFrame with added columns:
        car         - cumulative abnormal return
        label       - binary outcome variable
    
    Drops rows where CAR cannot be computed (insufficient history).
    """
    print("Computing abnormal returns for all earnings events...")
    print(f"  Using FF3 estimation window: {ESTIMATION_WINDOW[0]} to {ESTIMATION_WINDOW[1]} days")
    print(f"  Event window: {EVENT_WINDOW[0]} to +{EVENT_WINDOW[1]} days")
    print(f"  Label threshold: CAR > {threshold * 100:.1f}%")
    
    cars = []
    total = len(earnings)
    
    for i, row in earnings.iterrows():
        if i % 500 == 0:
            print(f"  Progress: {i}/{total}...")
        car = compute_abnormal_return(
            ticker=row["ticker"],
            earnings_date=row["earnings_date"],
            prices=prices,
            ff_factors=ff_factors,
        )
        cars.append(car)
    
    earnings = earnings.copy()
    earnings["car"] = cars
    earnings = earnings.dropna(subset=["car"])
    earnings["label"] = (earnings["car"] > threshold).astype(int)
    
    pos_rate = earnings["label"].mean()
    print(f"\nLabel construction complete.")
    print(f"  Total events: {len(earnings):,}")
    print(f"  Positive class (CAR > {threshold*100:.1f}%): {pos_rate:.1%}")
    print(f"  NOTE: Class imbalance is expected. Use precision-recall AUC,")
    print(f"        not accuracy, as the primary evaluation metric.")
    
    return earnings


# ── Stage 2: Feature Engineering ─────────────────────────────────────────────

def _get_prior_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the most recent trading day strictly before date."""
    prior = trading_days[trading_days < date]
    return prior[-1] if len(prior) > 0 else None


def compute_features_for_event(
    row: pd.Series,
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    macro: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> dict | None:
    """
    Compute the full feature vector for a single earnings event.
    
    ALL features use only information available strictly before earnings_date.
    The cutoff date is (earnings_date - 1 trading day).
    
    Feature groups:
        A. EPS surprise history  — past surprise patterns for this company
        B. Price & momentum      — pre-announcement price behavior
        C. Volatility            — realized volatility before announcement
        D. Fundamental           — valuation and size signals
        E. Macro regime          — market environment context
    
    Returns a dict of feature values, or None if insufficient data.
    """
    ticker = row["ticker"]
    earnings_date = row["earnings_date"]
    
    # Get this ticker's prices, sorted by date
    t_prices = prices[prices["ticker"] == ticker].set_index("date")["adj_close"].sort_index()
    trading_days = t_prices.index
    
    # Strict cutoff: all features use data up to but NOT including earnings_date
    cutoff = _get_prior_trading_day(earnings_date, trading_days)
    if cutoff is None:
        return None
    
    t_prices_pre = t_prices[t_prices.index <= cutoff]
    if len(t_prices_pre) < 60:
        return None  # Need at least 60 days of history
    
    # Daily log returns up to cutoff
    returns = np.log(t_prices_pre / t_prices_pre.shift(1)).dropna()
    
    features = {"ticker": ticker, "earnings_date": earnings_date}
    
    # ── A. EPS Surprise History ──────────────────────────────────────────────
    # Prior earnings events for this ticker, strictly before current event
    prior_eps = earnings[
        (earnings["ticker"] == ticker) &
        (earnings["earnings_date"] < earnings_date)
    ].sort_values("earnings_date")
    
    if len(prior_eps) >= 1:
        features["eps_surprise_last_q"] = prior_eps["eps_surprise_pct"].iloc[-1]
        features["eps_beat_last_q"] = float(prior_eps["eps_surprise_pct"].iloc[-1] > 0)
    else:
        features["eps_surprise_last_q"] = np.nan
        features["eps_beat_last_q"] = np.nan
    
    if len(prior_eps) >= 4:
        last4 = prior_eps["eps_surprise_pct"].iloc[-4:]
        features["eps_surprise_mean_4q"] = last4.mean()
        features["eps_surprise_std_4q"] = last4.std()
        features["eps_beat_streak"] = float(
            (prior_eps["eps_surprise_pct"].iloc[-3:] > 0).all()
        )
        # Trend: is surprise magnitude growing or shrinking?
        features["eps_surprise_trend"] = float(np.polyfit(range(4), last4.values, 1)[0])
    else:
        features["eps_surprise_mean_4q"] = np.nan
        features["eps_surprise_std_4q"] = np.nan
        features["eps_beat_streak"] = np.nan
        features["eps_surprise_trend"] = np.nan
    
    # ── B. Price & Momentum ──────────────────────────────────────────────────
    # Short-term drift: return in the 5 days before announcement
    # Pre-announcement drift is documented in the literature; large moves
    # may signal information leakage or positioning
    if len(returns) >= 5:
        features["ret_5d"] = returns.iloc[-5:].sum()
    else:
        features["ret_5d"] = np.nan
    
    # Medium-term momentum: return over 60 trading days (~3 months)
    if len(returns) >= 60:
        features["ret_60d"] = returns.iloc[-60:].sum()
    else:
        features["ret_60d"] = np.nan
    
    # Longer-term momentum: return over 252 trading days (~1 year)
    # Excludes most recent month to avoid short-term reversal contamination
    if len(returns) >= 252:
        features["ret_252d_ex1m"] = returns.iloc[-252:-21].sum()
    else:
        features["ret_252d_ex1m"] = np.nan
    
    # ── C. Volatility ────────────────────────────────────────────────────────
    # Realized volatility: annualized std of daily returns
    if len(returns) >= 30:
        features["realized_vol_30d"] = returns.iloc[-30:].std() * np.sqrt(252)
    else:
        features["realized_vol_30d"] = np.nan
    
    if len(returns) >= 60:
        features["realized_vol_60d"] = returns.iloc[-60:].std() * np.sqrt(252)
    else:
        features["realized_vol_60d"] = np.nan
    
    # Volatility ratio: is volatility rising or falling into the announcement?
    if len(returns) >= 60:
        vol_recent = returns.iloc[-10:].std()
        vol_longer = returns.iloc[-60:].std()
        features["vol_ratio"] = vol_recent / vol_longer if vol_longer > 0 else np.nan
    else:
        features["vol_ratio"] = np.nan
    
    # ── D. Fundamental ───────────────────────────────────────────────────────
    # Historical P/E = price at cutoff / trailing twelve month EPS
    # Both components are point-in-time accurate

    # Price at cutoff — already available in t_prices_pre
    price_at_cutoff = float(t_prices_pre.iloc[-1])

    # Trailing twelve month EPS = sum of last 4 reported quarterly EPS
    # Use prior_eps which is already filtered to earnings before this event
    if len(prior_eps) >= 4:
        ttm_eps = prior_eps["eps_actual"].iloc[-4:].sum()
        if ttm_eps > 0:  # only meaningful for profitable companies
            features["pe_ratio"] = price_at_cutoff / ttm_eps
        else:
            features["pe_ratio"] = np.nan
    else:
        features["pe_ratio"] = np.nan
    
    # ── E. Macro Regime ──────────────────────────────────────────────────────
    macro_pre = macro[macro["date"] <= cutoff].sort_values("date")
    
    if len(macro_pre) > 0:
        latest_macro = macro_pre.iloc[-1]
        features["vix_level"] = latest_macro.get("vix", np.nan)
        features["treasury_10y"] = latest_macro.get("treasury_10y", np.nan)
        features["yield_spread"] = latest_macro.get("yield_spread", np.nan)

        # VIX relative to its trailing 252-day median
        # Captures whether fear is elevated relative to recent norms
        # rather than using absolute VIX (which differs across regimes)
        if len(macro_pre) >= 252 and "vix" in macro_pre.columns:
            vix_median = macro_pre["vix"].iloc[-252:].median()
            features["vix_relative"] = (
                latest_macro["vix"] / vix_median if vix_median > 0 else np.nan
            )
        else:
            features["vix_relative"] = np.nan

        # Market return in the 10 days before announcement
        # Captures broad market backdrop at the time of announcement
        if len(macro_pre) >= 10:
            ff_pre = ff_factors[ff_factors["date"] <= cutoff].sort_values("date")
            if len(ff_pre) >= 10:
                features["mkt_ret_10d"] = ff_pre["mkt_rf"].iloc[-10:].sum()
            else:
                features["mkt_ret_10d"] = np.nan
        else:
            features["mkt_ret_10d"] = np.nan
    else:
        for col in ["vix_level", "treasury_10y", "yield_spread",
                    "vix_relative", "mkt_ret_10d"]:
            features[col] = np.nan
    
    return features


def build_features(
    labeled_earnings: pd.DataFrame,
    prices: pd.DataFrame,
    earnings: pd.DataFrame,
    macro: pd.DataFrame,
    ff_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature matrix for all labeled earnings events.
    
    Returns a DataFrame with one row per event containing:
        - ticker, earnings_date (identifiers)
        - All feature columns (Groups A-E above)
        - label, car (target variables — kept here for convenience)
    """
    print("Engineering features for all labeled events...")
    print("  All features use data strictly before the earnings announcement date.")
    print("  Lookahead bias is prevented by a strict date cutoff at each step.")
    
    all_features = []
    total = len(labeled_earnings)
    
    for i, row in labeled_earnings.iterrows():
        if i % 200 == 0:
            print(f"  Progress: {i}/{total}...")
        feat = compute_features_for_event(
            row=row,
            prices=prices,
            earnings=earnings,
            macro=macro,
            ff_factors=ff_factors,
        )
        if feat is not None:
            feat["label"] = row["label"]
            feat["car"] = row["car"]
            all_features.append(feat)
    
    feature_df = pd.DataFrame(all_features)
    feature_df["earnings_date"] = pd.to_datetime(feature_df["earnings_date"])
    feature_df = feature_df.sort_values("earnings_date").reset_index(drop=True)
    
    n_features = len([c for c in feature_df.columns
                       if c not in ["ticker", "earnings_date", "label", "car"]])
    print(f"\nFeature matrix built: {len(feature_df):,} events × {n_features} features")
    
    missing_pct = feature_df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.1]
    if len(high_missing) > 0:
        print("\nFeatures with >10% missing values:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.1%} missing")
    
    return feature_df


# ── Interaction Terms and Polynomial Features ─────────────────────────────────

FEATURE_COLS = [
    "eps_surprise_last_q", "eps_beat_last_q", "eps_surprise_mean_4q",
    "eps_surprise_std_4q", "eps_beat_streak", "eps_surprise_trend",
    "ret_5d", "ret_60d", "ret_252d_ex1m",
    "realized_vol_30d", "realized_vol_60d", "vol_ratio",
    "pe_ratio",
    "vix_level", "treasury_10y", "yield_spread",
    "vix_relative", "mkt_ret_10d",
]

# Economically motivated interaction terms
# Each tuple is (feature_A, feature_B) → product added as feature_A_x_feature_B
# Rationale documented inline
INTERACTION_PAIRS = [
    # Momentum is more informative in low-noise environments
    ("ret_5d", "vix_relative"),
    ("ret_60d", "vix_relative"),
    # EPS surprise history matters more in high-uncertainty periods
    ("eps_surprise_mean_4q", "realized_vol_30d"),
    # Market backdrop interacts with individual stock momentum
    ("ret_5d", "mkt_ret_10d"),
]

# Features where we expect nonlinear (quadratic) relationships
# Rationale: extreme values may predict mean reversion rather than continuation
POLYNOMIAL_FEATURES = ["ret_5d", "ret_60d", "vix_relative", "eps_surprise_last_q"]


def add_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms and polynomial features to the feature matrix.
    
    This is called BEFORE scaling in the modeling pipeline.
    The correct order is:
        raw features → add interactions/polynomials → scale → fit Lasso
    
    NOT:
        raw features → scale → add interactions/polynomials → Lasso
    
    The second ordering is wrong because squaring a standardized feature
    changes its meaning — X^2 after standardization is not the same as
    standardizing X^2.
    
    Lasso with L1 regularization will zero out interaction and polynomial
    terms that don't contribute marginal predictive value, so including
    them here is safe — the model self-selects.
    """
    df = df.copy()
    
    # Polynomial (quadratic) terms
    for feat in POLYNOMIAL_FEATURES:
        if feat in df.columns:
            df[f"{feat}_sq"] = df[feat] ** 2
    
    # Interaction terms
    for feat_a, feat_b in INTERACTION_PAIRS:
        if feat_a in df.columns and feat_b in df.columns:
            col_name = f"{feat_a}_x_{feat_b}"
            df[col_name] = df[feat_a] * df[feat_b]
    
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excludes identifiers and targets)."""
    exclude = {"ticker", "earnings_date", "label", "car"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    print("features.py loaded successfully.")
    print(f"Base features: {len(FEATURE_COLS)}")
    print(f"Interaction pairs: {len(INTERACTION_PAIRS)}")
    print(f"Polynomial features: {len(POLYNOMIAL_FEATURES)}")
    print(f"Total features (approx): {len(FEATURE_COLS) + len(INTERACTION_PAIRS) + len(POLYNOMIAL_FEATURES)}")
