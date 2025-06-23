# feature_build.py — build daily feature matrix + 63‑day excess‑return label
# ==============================================================================
# 1. Resamples quarterly fundamentals to *daily* (lagged by INFO_LAG_DAYS so you
#    never peek).
# 2. Joins with daily adjusted‑close prices & shares‑outstanding → market‑cap.
# 3. Computes basic ratios + 63‑day excess return vs SPY.
# 4. Writes data/features.parquet ready for model training.
#
# Assumes you have already run:
#   • bronze_clean.py  → data/bronze/*.parquet
#   • silver_join.py   → data/silver/fundamentals.parquet
#   • price_download.py (or sec_download.py) → price parquet files & shares.csv
# ==============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────
SILVER_FUND   = Path("data/silver/fundamentals.parquet")
PRICE_DIR     = Path("data")
SHARES_CSV    = PRICE_DIR / "shares.csv"
FEATURES_OUT  = Path("data/features.parquet")

INFO_LAG_DAYS = 45      # filing usable 45 d after quarter‑end
MAX_FF_DAYS   = 400      # forward-fill fundamentals roughly 1 year
      # forward‑fill fundamentals ≤ one quarter

TICKER_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
}

# ── LOAD FUNDAMENTALS ────────────────────────────────────────────────
print("📥  fundamentals …")
fund = pd.read_parquet(SILVER_FUND)
fund["end"] = pd.to_datetime(fund["end"], utc=True)
fund["info_date"] = fund["end"] + pd.Timedelta(days=INFO_LAG_DAYS)

# ── LOAD SHARES OUTSTANDING ──────────────────────────────────────────
shares_map = (
    pd.read_csv(SHARES_CSV)
      .set_index("ticker")["shares_outstanding"].to_dict()
)

# ── PRICE HELPERS ────────────────────────────────────────────────────

def adj_close(df: pd.DataFrame) -> pd.Series:
    """Return the adjusted‑close column from a yfinance‑style frame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(str(c) for c in tup if c) for tup in df.columns]
    for col in df.columns:
        low = col.lower()
        if "adj" in low and "close" in low:
            return df[col]
    for col in df.columns:
        if "close" in col.lower():
            return df[col]
    raise KeyError("No close column found in price frame")

# ── LOAD PRICE SERIES ────────────────────────────────────────────────
print("📥  prices …")
price = {}
for tkr in TICKER_CIK:
    p = pd.read_parquet(PRICE_DIR / f"{tkr}.parquet")
    p.index = pd.to_datetime(p.index, utc=True)
    price[tkr] = adj_close(p).rename("adj_close").to_frame()

spy_raw = pd.read_parquet(PRICE_DIR / "SPY.parquet")
spy = adj_close(spy_raw).rename("spy_close").to_frame()
spy.index = pd.to_datetime(spy.index, utc=True)
spy = spy.asfreq("D").ffill()

# ── FUNDAMENTALS → DAILY ─────────────────────────────────────────────
parts = []
for cik, g in fund.groupby("cik"):
    daily = (
        g.set_index("info_date")
         .asfreq("D")
         .ffill(limit=MAX_FF_DAYS)
         .assign(cik=cik)
    )
    parts.append(daily)

fund_daily = pd.concat(parts).reset_index(names="date")

# ── MERGE PRICE + FUNDAMENTALS + MARKET CAP ──────────────────────────
features = []
for tkr, cik in TICKER_CIK.items():
    cik_clean = cik.lstrip("0")  # remove leading zeros for matching
    f = fund_daily[fund_daily["cik"].str.lstrip("0") == cik_clean]
    p = price[tkr].reset_index(names="date")
    merged = p.merge(f, on="date", how="left")

    shares = shares_map.get(tkr)
    merged["market_cap"] = (
        np.nan if shares in (None, "NA") else merged["adj_close"] * float(shares)
    )
    merged["market_cap_log"] = np.log1p(merged["market_cap"])
    merged["ticker"] = tkr
    features.append(merged)

features = pd.concat(features, ignore_index=True)

# ── KEEP ONLY ROWS WITH VALID, POSITIVE EQUITY ───────────────────────
features = features[features["equity"].notna() & (features["equity"] > 0)]

# ── DERIVED RATIOS ───────────────────────────────────────────────────
eps = 1e-9
features["debt_equity"]  = features["liabilities"] / (features["equity"] + eps)
features["gross_margin"] = 1 - (features["cogs"] / (features["revenue"] + eps))
features["rev_qoq"] = (
    features.groupby("ticker")["revenue"]
            .transform(lambda x: x.pct_change(periods=MAX_FF_DAYS, fill_method=None))
)


# ── TARGET LABEL (63‑day excess return) ───────────────────────────────
features = features.merge(spy, left_on="date", right_index=True, how="left")
for col in ["adj_close", "spy_close"]:
    features[f"fwd_{col}"] = features.groupby("ticker")[col].shift(-63)

features["future_return"]  = (features["fwd_adj_close"] / features["adj_close"]) - 1
features["spy_future_ret"] = (features["fwd_spy_close"] / features["spy_close"]) - 1
features["excess_ret"]     = features["future_return"] - features["spy_future_ret"]
features["label_up"]       = (features["excess_ret"] > 0).astype(int)
features.drop(columns=["fwd_adj_close", "fwd_spy_close"], inplace=True)

# ── SAVE ─────────────────────────────────────────────────────────────
FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
features.to_parquet(FEATURES_OUT)
print(f"✅  wrote {FEATURES_OUT}  → {len(features):,} rows, {features.ticker.nunique()} tickers")
