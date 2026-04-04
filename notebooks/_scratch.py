import pandas as pd
df = pd.read_parquet("data/processed/feature_matrix.parquet")
#print(df[["log_mkt_cap", "ret_5d", "realized_vol_30d"]].describe())
df["mkt_cap_decile"] = pd.qcut(df["log_mkt_cap"], q=10, labels=False)
#print(df.groupby("mkt_cap_decile")["ticker"].nunique())
# Look at the correlation between log_mkt_cap and your label
print(df.groupby("label")["log_mkt_cap"].mean())