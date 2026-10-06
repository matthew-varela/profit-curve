#!/usr/bin/env python3
"""Download daily OHLCV prices and shares outstanding for the ticker universe."""

from __future__ import annotations

import argparse

import yfinance as yf

from feature_config import BENCHMARK_TICKER, PRICE_DIR, START_DATE, TICKER_CIK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV data")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Optional subset of tickers. Defaults to configured expanded universe.",
    )
    parser.add_argument("--start", default=START_DATE, help="Download start date")
    parser.add_argument(
        "--include-benchmark",
        action="store_true",
        default=True,
        help="Also download the SPY benchmark parquet",
    )
    return parser.parse_args()


def download_prices(tickers: list[str], start: str) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    for tkr in tickers:
        print(f"Downloading {tkr} OHLCV prices...")
        df = yf.download(tkr, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"WARNING: {tkr} returned no price rows")
            continue

        # Keep the full yfinance OHLCV frame so feature_build can derive range,
        # gap, close-position, volatility, and volume features.
        df.to_parquet(PRICE_DIR / f"{tkr}.parquet")
        print(f"Saved {PRICE_DIR / f'{tkr}.parquet'} ({len(df):,} rows)")


def download_shares(tickers: list[str]) -> None:
    with open(PRICE_DIR / "shares.csv", "w") as f:
        f.write("ticker,shares_outstanding\n")
        for tkr in tickers:
            ticker_obj = yf.Ticker(tkr)

            shares = ticker_obj.fast_info.get("sharesOutstanding")
            if shares in (None, 0):
                info = ticker_obj.get_info()
                shares = info.get("sharesOutstanding") or info.get("sharesOutstandingPrevious")
                if shares is None:
                    print(f"WARNING: {tkr}: sharesOutstanding missing; writing NA")
                    shares = "NA"

            f.write(f"{tkr},{shares}\n")
            if isinstance(shares, (int, float)):
                print(f"Saved {tkr} shares outstanding: {shares:,}")
            else:
                print(f"Saved {tkr} shares outstanding: {shares}")


def main() -> None:
    args = parse_args()
    tickers = [t.upper() for t in args.tickers] if args.tickers else list(TICKER_CIK)
    price_symbols = tickers + ([BENCHMARK_TICKER] if args.include_benchmark else [])

    download_prices(price_symbols, args.start)
    download_shares(tickers)


if __name__ == "__main__":
    main()
