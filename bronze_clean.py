# ====================================================================
# This version fixes missing-equity NaNs by mapping **all common equity XBRL tags**
# to a single canonical column and restores the safe path-printing that falls back
# to an absolute path when the file isn't a sub-path of the current working dir.
# ====================================================================

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
BRONZE_DIR = Path("data/bronze")

# Map canonical column → list of possible SEC tags
TAG_MAP: Dict[str, List[str]] = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToParent",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "TotalEquityGross",
    ],
    "revenue": ["Revenues"],
    "cogs": ["CostOfRevenue"],
    "net_income": ["NetIncomeLoss"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["CapitalExpenditures"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "accounts_payable": ["AccountsPayableCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "interest_expense": ["InterestExpenseNonOperating", "InterestExpense"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization"],
    "intangibles": ["FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill"],
    "goodwill": ["Goodwill"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

KEEP_PERIODS = {"Q1", "Q2", "Q3", "Q4", "FY"}

# ── Helpers ──────────────────────────────────────────────────────────

def first_numeric_series(us_gaap: dict, tag_list: List[str], unit: str = "USD") -> List[dict] | None:
    for tag in tag_list:
        series = (
            us_gaap.get(tag, {})
                   .get("units", {})
                   .get(unit)
        )
        if series:
            return series
    return None


def extract_company_table(json_path: Path) -> pd.DataFrame:
    blob = json.loads(json_path.read_text())
    cik = blob["cik"]
    us_gaap = blob["facts"].get("us-gaap", {})

    rows = []
    for col, tags in TAG_MAP.items():
        series = first_numeric_series(us_gaap, tags)
        if not series:
            continue
        for rec in series:
            if rec["fp"] not in KEEP_PERIODS:
                continue
            rows.append({
                "cik": cik,
                "end": rec["end"],
                "fy": rec["fy"],
                "fp": rec["fp"],
                "concept": col,
                "value": rec["val"],
            })

    if not rows:
        return pd.DataFrame()

    df_long = pd.DataFrame(rows)
    df_long = (
        df_long.sort_values("end")
               .drop_duplicates(["cik", "fy", "fp", "concept"], keep="last")
    )

    df_wide = (
        df_long.pivot_table(index=["cik", "end", "fy", "fp"],
                             columns="concept", values="value")
                .reset_index()
                .sort_values("end")
    )
    df_wide.columns.name = None
    return df_wide


def process_companies(files: Iterable[Path]) -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    for path in files:
        print(f"🔄  Cleaning {path.name} …")
        df = extract_company_table(path)
        if df.empty:
            print(f"⚠️  No qualifying data in {path.name}; skipping")
            continue
        out = BRONZE_DIR / f"{path.stem}.parquet"
        df.to_parquet(out)
        try:
            rel = out.resolve().relative_to(Path.cwd())
        except ValueError:
            rel = out.resolve()  # fall back to absolute path when out isn't sub‑path
        print(f"✅  {rel} written ({len(df)} rows)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        wanted = {a.upper().lstrip("0") for a in args}
        files = [p for p in RAW_DIR.glob("*.json") if p.stem.lstrip("0") in wanted]
    else:
        files = list(RAW_DIR.glob("*.json"))

    if not files:
        print("No raw JSON files match selection — nothing to do.")
        sys.exit(0)

    process_companies(files)
