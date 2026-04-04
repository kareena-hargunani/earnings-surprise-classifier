# Once I have the labeled dataset, I want to measure the spillover effect of the release of earnings reports in the same week. 
# For each earnings event, count how many same-sector companies
# announced earnings within a 5-day window
# If spillover is large, CAR should be higher when this count is high
import pandas as pd
df = pd.read_parquet("data/raw/earnings.parquet")
print(df.columns.tolist())

df["sector"] = df["ticker"].map(ticker_sector_map)
df["earnings_week"] = df["earnings_date"].dt.to_period("W")

sector_week_counts = (
    df.groupby(["sector", "earnings_week"])
    .size()
    .reset_index(name="same_sector_announcements")
)

df = df.merge(sector_week_counts, on=["sector", "earnings_week"])

# Now look at whether CAR correlates with same_sector_announcements
print(df.groupby(
    pd.qcut(df["same_sector_announcements"], q=3, labels=["low", "medium", "high"])
)["car"].mean())