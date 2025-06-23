#!/usr/bin/env python3
"""
Download SEC `companyfacts` JSON for provided tickers/CIKs and store benchmark
price history (default: SPY).

Examples
--------
python sec_download.py AAPL MSFT
python sec_download.py 0000320193
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Iterable

import requests
import yfinance as yf

# ── Config ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PRICE_DIR = DATA_DIR
PRICE_SYMS = ["SPY"]

HEADERS = {
    "User-Agent": "MarketPredictor/0.1 (matthewvarela8@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}

SEC_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
MAX_RETRIES = 5
MAX_RPS = 10
SLEEP_BETWEEN_OK = 1 / MAX_RPS


# ── Helpers ──────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)


def download_price_history(symbols: Iterable[str] = PRICE_SYMS, start: str = "2010-01-01") -> None:
    """Grab benchmark OHLCV (auto-adjusted) and save to parquet."""
    for sym in symbols:
        print(f"📈  Fetching {sym} price history …")
        df = yf.download(sym, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"⚠️   No data returned for {sym}")
            continue
        out = PRICE_DIR / f"{sym}.parquet"
        df.to_parquet(out)
        try:
            rel = out.resolve().relative_to(Path.cwd())
        except ValueError:
            rel = out.resolve()
        print(f"✅  {sym} → {rel}")


def fetch_company_facts(ciks: Iterable[str]) -> None:
    """Download companyfacts JSON for each CIK with retry/back-off."""
    session = requests.Session()
    session.headers.update(HEADERS)

    for raw in ciks:
        cik = str(raw).lstrip("0").zfill(10)
        url = SEC_URL_TMPL.format(cik=cik)

        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(url, timeout=30)
                status = r.status_code
            except requests.RequestException as exc:
                print(f"❌  {cik} network error: {exc}")
                status = None

            if status == 200:
                out = RAW_DIR / f"{cik}.json"
                out.write_bytes(r.content)
                print(f"✅  {cik} saved ({len(r.content):,} bytes)")
                time.sleep(SLEEP_BETWEEN_OK)
                break

            if status == 404:
                print(f"🚫  {cik} not found (404)")
                break

            wait = (2 ** attempt) + random.random()
            print(f"🔄  {cik} HTTP {status} — retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
        else:
            print(f"❌  {cik} FAILED after {MAX_RETRIES} attempts")


def ticker_to_cik(ticker: str) -> str | None:
    """Resolve ticker → 10-digit CIK using three fallbacks."""
    tkr = ticker.upper()

    try:
        info = yf.Ticker(tkr).fast_info or yf.Ticker(tkr).get_info()
        if info and info.get("cik"):
            return str(info["cik"]).zfill(10)
    except Exception:
        pass

    try:
        mapping = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30).json()
        for rec in mapping.values():
            if rec["ticker"].upper() == tkr:
                return str(rec["cik_str"]).zfill(10)
    except Exception:
        pass

    try:
        txt = requests.get("https://www.sec.gov/include/ticker.txt", headers=HEADERS, timeout=30).text
        for line in txt.strip().splitlines():
            sym, cik = line.split("|")
            if sym.upper() == tkr:
                return str(cik).zfill(10)
    except Exception:
        pass

    return None


# ── CLI / Entry-point ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEC companyfacts downloader")
    p.add_argument("identifiers", nargs="+", help="Tickers or 10-digit CIKs")
    return p.parse_args()


def main() -> None:
    ensure_dirs()

    ids = parse_args().identifiers
    ciks: list[str] = []

    for ident in ids:
        if ident.isdigit():
            ciks.append(ident)
        else:
            cik = ticker_to_cik(ident)
            if cik:
                ciks.append(cik)
                print(f"🔎  {ident.upper()} → {cik}")
            else:
                print(f"⚠️   Could not resolve {ident}")

    if not ciks:
        print("Nothing to download.")
        return

    download_price_history()
    fetch_company_facts(ciks)


if __name__ == "__main__":
    main()
