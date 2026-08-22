"""
data.py
-------
Data collection for the Earnings Surprise Classifier project.

Pulls three data sources:
  1. S&P 500 constituent list (Wikipedia)
  2. Daily adjusted close prices + volume (yfinance)
  3. Earnings dates + EPS estimates vs actuals (yfinance)
  4. Fama-French 3-factor data (Ken French data library)
  5. Macro regime indicators (FRED via pandas_datareader)

All raw data is cached to data/raw/ as CSV files immediately after
pulling to avoid repeated API calls. Always load from cache first.

IMPORTANT: This module never uses future information. All pulls are
for historical data only. Lookahead bias is enforced at the feature
engineering stage (features.py), not here.
"""

import os
import time
import warnings
import urllib.request
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── Universe ────────────────────────────────────────────────────────────────

def get_sp500_tickers(cache: bool = True) -> list[str]:
    """
    Pull current S&P 500 constituents from Wikipedia.
    
    NOTE: This uses the *current* S&P 500 list, which introduces survivorship
    bias — companies that were delisted or removed are excluded. This is a
    known limitation documented in the project write-up. For a production
    system you would use a point-in-time constituent list from CRSP or
    Compustat.
    
    Returns a list of ticker strings, e.g. ['AAPL', 'MSFT', ...]
    """
    cache_path = os.path.join(RAW_DIR, "sp500_tickers.csv")
    
    if cache and os.path.exists(cache_path):
        print("Loading S&P 500 tickers from cache...")
        return pd.read_csv(cache_path)["ticker"].tolist()
    
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia rejects requests without a browser-like User-Agent (403).
    # Fetch manually rather than via read_html's storage_options, which
    # isn't supported on all pandas versions.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read()
    tables = pd.read_html(html)
    df = tables[0][["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    )
    # Wikipedia uses '.' in some tickers; yfinance uses '-'
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} tickers to cache.")
    return df["ticker"].tolist()


# ── Price Data ───────────────────────────────────────────────────────────────

def get_price_data(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Pull daily adjusted close prices for a list of tickers.
    
    Returns a DataFrame with columns [date, ticker, adj_close, volume].
    Cached as a single parquet file for efficiency.
    
    Parameters
    ----------
    tickers : list of ticker strings
    start   : start date string 'YYYY-MM-DD'
    end     : end date string; defaults to today
    cache   : load from cache if available
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_path = os.path.join(RAW_DIR, "prices.parquet")
    
    if cache and os.path.exists(cache_path):
        print("Loading price data from cache...")
        return pd.read_parquet(cache_path)
    
    print(f"Fetching price data for {len(tickers)} tickers ({start} to {end})...")
    print("  This may take several minutes — data is cached after first pull.")
    
    # Pull in batches of 50 to avoid rate limiting
    batch_size = 50
    all_prices = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{len(tickers) // batch_size + 1}...")
        try:
            raw = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            # yfinance returns MultiIndex columns when multiple tickers
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"].stack().reset_index()
                closes.columns = ["date", "ticker", "adj_close"]
                volumes = raw["Volume"].stack().reset_index()
                volumes.columns = ["date", "ticker", "volume"]
                batch_df = closes.merge(volumes, on=["date", "ticker"])
            else:
                # Single ticker fallback
                batch_df = raw[["Close", "Volume"]].reset_index()
                batch_df.columns = ["date", "adj_close", "volume"]
                batch_df["ticker"] = batch[0]
            all_prices.append(batch_df)
        except Exception as e:
            print(f"  Warning: failed batch {i // batch_size + 1}: {e}")
        time.sleep(1)  # polite rate limiting
    
    prices = pd.concat(all_prices, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_parquet(cache_path, index=False)
    print(f"  Cached {len(prices):,} price rows.")
    return prices


# ── Earnings Data ────────────────────────────────────────────────────────────

def get_earnings_data(
    tickers: list[str],
    cache: bool = True,
) -> pd.DataFrame:
    """
    Pull quarterly earnings dates, EPS estimates, and EPS actuals for each ticker.
    
    Returns a DataFrame with columns:
        [ticker, earnings_date, eps_estimate, eps_actual, eps_surprise_pct]
    
    eps_surprise_pct = (actual - estimate) / |estimate|
    
    NOTE: yfinance earnings data quality is imperfect. Some tickers will have
    missing estimates. These are dropped rather than imputed to avoid introducing
    noise into the label construction. If you have WRDS/IBES access, replacing
    this function with an IBES pull will significantly improve data quality.
    """
    cache_path = os.path.join(RAW_DIR, "earnings.parquet")
    
    if cache and os.path.exists(cache_path):
        print("Loading earnings data from cache...")
        return pd.read_parquet(cache_path)
    
    print(f"Fetching earnings data for {len(tickers)} tickers...")
    print("  This pulls one ticker at a time — expect ~10-15 minutes for full S&P 500.")
    
    records = []
    failed = []
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tickers)} tickers...")
        try:
            t = yf.Ticker(ticker)
            # The .earnings_dates property wraps get_earnings_dates() with a
            # low implicit limit (empirically ~24 quarters / 6 years), which
            # doesn't reach back to START_DATE. Calling get_earnings_dates()
            # directly with an explicit limit pulls much deeper history —
            # verified back to 2005+ with usable EPS estimate/actual data.
            # limit=60 (~15 years) comfortably covers START_DATE plus a
            # buffer for the trailing-4-quarter EPS-history features.
            earnings = t.get_earnings_dates(limit=60)
            if earnings is None or earnings.empty:
                continue
            earnings = earnings.reset_index()
            earnings.columns = [c.strip() for c in earnings.columns]
            earnings["ticker"] = ticker
            earnings = earnings.rename(columns={
                "Earnings Date": "earnings_date",
                "EPS Estimate": "eps_estimate",
                "Reported EPS": "eps_actual",
            })
            # Drop rows where either estimate or actual is missing
            earnings = earnings.dropna(subset=["eps_estimate", "eps_actual"])
            if earnings.empty:
                continue
            # Calculate surprise percentage
            earnings["eps_surprise_pct"] = (
                (earnings["eps_actual"] - earnings["eps_estimate"])
                / earnings["eps_estimate"].abs()
            )
            records.append(earnings[["ticker", "earnings_date", "eps_estimate",
                                      "eps_actual", "eps_surprise_pct"]])
        except Exception as e:
            failed.append(ticker)
        time.sleep(0.3)
    
    if failed:
        print(f"  Warning: failed to fetch earnings for {len(failed)} tickers: {failed[:10]}...")
    
    df = pd.concat(records, ignore_index=True)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df):,} earnings observations.")
    return df


# ── Fama-French Factors ──────────────────────────────────────────────────────

def get_fama_french_factors(
    start: str = "2015-01-01",
    end: str = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Pull Fama-French 3-factor daily data from Ken French's data library.
    
    Returns a DataFrame with columns:
        [date, mkt_rf, smb, hml, rf]
    
    All values are in decimal form (divided by 100 from the raw percentage).
    
    These factors are used in label construction (computing abnormal returns)
    and can also be used as features (market environment context).
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_path = os.path.join(RAW_DIR, "fama_french.csv")
    
    if cache and os.path.exists(cache_path):
        print("Loading Fama-French factors from cache...")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df
    
    print("Fetching Fama-French 3-factor data from Ken French library...")
    try:
        ff = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start=start,
            end=end,
        )[0]
        ff = ff / 100  # convert from percentage to decimal
        ff = ff.reset_index()
        ff.columns = ["date", "mkt_rf", "smb", "hml", "rf"]
        ff["date"] = pd.to_datetime(ff["date"])
        ff.to_csv(cache_path, index=False)
        print(f"  Cached {len(ff):,} trading days of FF factors.")
        return ff
    except Exception as e:
        print(f"  Error fetching FF factors: {e}")
        print("  Try: pip install pandas_datareader")
        raise


# ── FRED Macro Regime Indicators ─────────────────────────────────────────────

def get_macro_indicators(
    start: str = "2015-01-01",
    end: str = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Pull macro regime indicator series from FRED.
    
    Series pulled:
        VIXCLS  - CBOE VIX daily close
        DGS10   - 10-year Treasury yield
        T10Y2Y  - 10yr minus 2yr yield spread (yield curve)
        USREC   - NBER recession indicator (monthly, forward-filled to daily)
    
    Returns a daily DataFrame with columns:
        [date, vix, treasury_10y, yield_spread, recession_indicator]
    
    These are used as features (regime context) not just fixed effects.
    They allow the model to condition predictions on the macro environment,
    partially addressing the cross-sectional dependence problem discussed
    in the project write-up.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_path = os.path.join(RAW_DIR, "macro_indicators.csv")
    
    if cache and os.path.exists(cache_path):
        print("Loading macro indicators from cache...")
        return pd.read_csv(cache_path, parse_dates=["date"])
    
    print("Fetching macro indicators from FRED...")
    series_map = {
        "VIXCLS": "vix",
        "DGS10": "treasury_10y",
        "T10Y2Y": "yield_spread",
        "USREC": "recession_indicator",
    }
    
    dfs = []
    for fred_code, col_name in series_map.items():
        try:
            s = web.DataReader(fred_code, "fred", start=start, end=end)
            s.columns = [col_name]
            dfs.append(s)
            print(f"  Fetched {fred_code}")
        except Exception as e:
            print(f"  Warning: could not fetch {fred_code}: {e}")
    
    macro = pd.concat(dfs, axis=1).reset_index()
    macro.columns = ["date"] + list(macro.columns[1:])
    macro["date"] = pd.to_datetime(macro["date"])
    
    # Forward-fill recession indicator (monthly series → daily)
    if "recession_indicator" in macro.columns:
        macro["recession_indicator"] = macro["recession_indicator"].ffill()
    
    # Forward-fill any other gaps from market holidays
    macro = macro.ffill()
    macro.to_csv(cache_path, index=False)
    print(f"  Cached {len(macro):,} rows of macro indicators.")
    return macro


# ── Convenience loader ───────────────────────────────────────────────────────

def load_all_data(
    n_tickers: int = 100,
    start: str = "2015-01-01",
    cache: bool = True,
) -> dict:
    """
    Convenience function that loads all data sources and returns them
    as a dictionary. Limits to n_tickers for development; use None
    for the full S&P 500.
    
    Usage:
        data = load_all_data(n_tickers=50)
        prices   = data['prices']
        earnings = data['earnings']
        ff       = data['fama_french']
        macro    = data['macro']
    """
    tickers = get_sp500_tickers(cache=cache)
    if n_tickers:
        tickers = tickers[:n_tickers]
        print(f"Using {n_tickers} tickers for development run.")
    
    return {
        "tickers": tickers,
        "prices": get_price_data(tickers, start=start, cache=cache),
        "earnings": get_earnings_data(tickers, cache=cache),
        "fama_french": get_fama_french_factors(start=start, cache=cache),
        "macro": get_macro_indicators(start=start, cache=cache),
    }


if __name__ == "__main__":
    # Quick smoke test — pulls 10 tickers only
    data = load_all_data(n_tickers=10, cache=False)
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {len(v)} items")
