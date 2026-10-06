#!/usr/bin/env python3
"""Download SEC companyfacts plus recent 10-K/10-Q filing text."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Iterable

import requests
import yfinance as yf
from bs4 import BeautifulSoup

from feature_config import (
    BENCHMARK_TICKER,
    FILINGS_DIR,
    PRICE_DIR,
    RAW_DIR,
    START_DATE,
    TICKER_CIK,
)

HEADERS = {
    "User-Agent": "MarketPredictor/0.1 (matthewvarela8@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}

SEC_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{primary_doc}"
MAX_RETRIES = 5
MAX_RPS = 10
SLEEP_BETWEEN_OK = 1 / MAX_RPS
FILING_FORMS = {"10-K", "10-Q"}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)


def download_price_history(symbols: Iterable[str] = (BENCHMARK_TICKER,), start: str = START_DATE) -> None:
    """Grab benchmark OHLCV (auto-adjusted) and save to parquet."""
    for sym in symbols:
        print(f"Fetching {sym} price history...")
        df = yf.download(sym, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"WARNING: no data returned for {sym}")
            continue
        out = PRICE_DIR / f"{sym}.parquet"
        df.to_parquet(out)
        try:
            rel = out.resolve().relative_to(Path.cwd())
        except ValueError:
            rel = out.resolve()
        print(f"Saved {sym} -> {rel}")


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
                print(f"ERROR: {cik} network error: {exc}")
                status = None

            if status == 200:
                out = RAW_DIR / f"{cik}.json"
                out.write_bytes(r.content)
                print(f"Saved {cik} companyfacts ({len(r.content):,} bytes)")
                time.sleep(SLEEP_BETWEEN_OK)
                break

            if status == 404:
                print(f"WARNING: {cik} not found (404)")
                break

            wait = (2 ** attempt) + random.random()
            print(f"Retrying {cik} HTTP {status}: {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
        else:
            print(f"ERROR: {cik} failed after {MAX_RETRIES} attempts")


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def fetch_filing_texts(ciks: Iterable[str], limit_per_cik: int = 8) -> None:
    """Download recent 10-K/10-Q filing documents as plain text for NLP."""
    session = requests.Session()
    session.headers.update(HEADERS)
    manifest = FILINGS_DIR / "manifest.csv"
    manifest_exists = manifest.exists()

    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cik", "form", "filing_date", "accession", "primary_doc", "path"],
        )
        if not manifest_exists:
            writer.writeheader()

        for raw in ciks:
            cik = str(raw).lstrip("0").zfill(10)
            submissions_url = SUBMISSIONS_URL_TMPL.format(cik=cik)
            try:
                sub = session.get(submissions_url, timeout=30)
                sub.raise_for_status()
                recent = sub.json()["filings"]["recent"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                print(f"WARNING: could not read submissions for {cik}: {exc}")
                continue

            selected = []
            for form, filing_date, accession, primary_doc in zip(
                recent.get("form", []),
                recent.get("filingDate", []),
                recent.get("accessionNumber", []),
                recent.get("primaryDocument", []),
            ):
                if form in FILING_FORMS:
                    selected.append((form, filing_date, accession, primary_doc))
                if len(selected) >= limit_per_cik:
                    break

            cik_dir = FILINGS_DIR / cik
            cik_dir.mkdir(parents=True, exist_ok=True)
            for form, filing_date, accession, primary_doc in selected:
                accession_clean = accession.replace("-", "")
                out = cik_dir / f"{filing_date}_{form.replace('-', '')}_{accession_clean}.txt"
                if out.exists():
                    continue

                url = ARCHIVES_URL_TMPL.format(
                    cik_int=int(cik),
                    accession=accession_clean,
                    primary_doc=primary_doc,
                )
                try:
                    response = session.get(url, timeout=45)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    print(f"WARNING: could not download filing {cik} {accession}: {exc}")
                    continue

                out.write_text(html_to_text(response.content), encoding="utf-8")
                writer.writerow(
                    {
                        "cik": cik,
                        "form": form,
                        "filing_date": filing_date,
                        "accession": accession,
                        "primary_doc": primary_doc,
                        "path": str(out),
                    }
                )
                print(f"Saved filing text {out}")
                time.sleep(SLEEP_BETWEEN_OK)


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
    p = argparse.ArgumentParser(description="SEC companyfacts and filing text downloader")
    p.add_argument(
        "identifiers",
        nargs="*",
        help="Tickers or CIKs. Defaults to the configured expanded ticker universe.",
    )
    p.add_argument("--skip-benchmark", action="store_true", help="Do not download SPY")
    p.add_argument("--skip-facts", action="store_true", help="Do not download companyfacts JSON")
    p.add_argument("--skip-filings", action="store_true", help="Do not download filing text")
    p.add_argument("--filing-limit", type=int, default=8, help="Recent 10-K/10-Q docs per CIK")
    return p.parse_args()


def main() -> None:
    ensure_dirs()

    args = parse_args()
    ids = args.identifiers or list(TICKER_CIK)
    ciks: list[str] = []

    for ident in ids:
        if ident.isdigit():
            ciks.append(ident.zfill(10))
        elif ident.upper() in TICKER_CIK:
            cik = TICKER_CIK[ident.upper()]
            ciks.append(cik)
            print(f"Resolved {ident.upper()} -> {cik}")
        else:
            cik = ticker_to_cik(ident)
            if cik:
                ciks.append(cik)
                print(f"Resolved {ident.upper()} -> {cik}")
            else:
                print(f"WARNING: could not resolve {ident}")

    if not ciks:
        print("Nothing to download.")
        return

    if not args.skip_benchmark:
        download_price_history()
    if not args.skip_facts:
        fetch_company_facts(ciks)
    if not args.skip_filings:
        fetch_filing_texts(ciks, limit_per_cik=args.filing_limit)


if __name__ == "__main__":
    main()
