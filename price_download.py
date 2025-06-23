#!/usr/bin/env python3
"""
price_download.py — grab daily prices & shares-outstanding for each ticker.
"""

from pathlib import Path
import yfinance as yf

TICKERS = {"AAPL": "0000320193", "MSFT": "0000789019"}
PRICE_DIR = Path("data")

PRICE_DIR.mkdir(parents=True, exist_ok=True)

# download OHLCV
for tkr in TICKERS:
    print(f"📈  Downloading {tkr} prices …")
    df = yf.download(tkr, start="2010-01-01", auto_adjust=True, progress=False)
    df.to_parquet(PRICE_DIR / f"{tkr}.parquet")

# one-time grab of current shares-outstanding
with open(PRICE_DIR / "shares.csv", "w") as f:
    f.write("ticker,shares_outstanding\n")
    for tkr in TICKERS:
        ticker_obj = yf.Ticker(tkr)

        # 1️⃣ fast_info first (quick, but may miss some fields)
        shares = ticker_obj.fast_info.get("sharesOutstanding")

        # 2️⃣ fall back to get_info() if None
        if shares in (None, 0):
            info = ticker_obj.get_info()
            shares = info.get("sharesOutstanding") or info.get("sharesOutstandingPrevious")  # extra fallback
            if shares is None:
                print(f"⚠️   {tkr}: sharesOutstanding missing — writing NA")
                shares = "NA"

        f.write(f"{tkr},{shares}\n")
        print(f"✅  {tkr} shares outstanding: {shares}")

        # Log the shares-outstanding value. Use thousands-separator formatting only
        # when we actually have a numeric value; the SEC or Yahoo endpoints can
        # return "NA" when the field is unavailable, which would raise a
        # ValueError with the ":," format spec.  We handle both cases cleanly
        # and avoid the duplicate print.
        if isinstance(shares, (int, float)):
            print(f"✅  {tkr} shares outstanding: {shares:,}")
        else:
            print(f"✅  {tkr} shares outstanding: {shares}")
