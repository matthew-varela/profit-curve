# silver_join.py — combine per‑company bronze tables into a master "silver" set
# ============================================================================
# This stage merges **all** quarterly fundamentals into a single Parquet file
# that downstream notebooks & models can load in one line.
#
# Why a separate script?
#   • Keeps join/ratio logic isolated (easy to rerun when bronze updates)
#   • Lets you swap pandas ↔ polars ↔ DuckDB without touching download/clean.
#   • Makes unit‑testing simple: input = a few bronze files; output = one table.
#
# What it does right now
#   1. Reads every Parquet in data/bronze/  (produced by bronze_clean.py)
#   2. Outer‑concats them, sorts chronologically
#   3. Optionally forward‑fills missing quarters (*disabled by default*)
#   4. Writes data/silver/fundamentals.parquet
#
# You can extend this script later to:
#   • join daily price & shares‑outstanding → compute MarketCap, EV/EBITDA, etc.
#   • add macro benchmarks
#   • calculate rolling YoY deltas (e.g., revenue_growth)
#
# Run:  python silver_join.py            # merge all bronze companies
#     or python silver_join.py AAPL MSFT  # subset by ticker/CIK
# ============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────
BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
OUTPUT_FILE = SILVER_DIR / "fundamentals.parquet"

# Forward‑fill fundamentals between quarters?  False keeps original dates only.
FILL_MISSING = False

# ── Helper functions ─────────────────────────────────────────────────

def read_bronze(file: Path) -> pd.DataFrame:
    """Load a per‑company parquet and ensure dtype consistency."""
    df = pd.read_parquet(file)
    # Ensure key columns are typed uniformly
    df["cik"] = df["cik"].astype(str)
    df["end"] = pd.to_datetime(df["end"])
    return df

# ── Main flow ───────────────────────────────────────────────────────

def gather_files(selection: List[str] | None = None) -> List[Path]:
    """Return list of bronze parquet paths, optionally filtered by ticker/CIK."""
    if selection:
        wanted = {s.upper().lstrip("0") for s in selection}
        return [p for p in BRONZE_DIR.glob("*.parquet") if p.stem.lstrip("0") in wanted]
    return list(BRONZE_DIR.glob("*.parquet"))


def main(argv: list[str]) -> None:
    files = gather_files(argv)
    if not files:
        print("No bronze files match selection — nothing to merge.")
        return

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for fp in files:
        print(f"📥  Reading {fp.name} …")
        dfs.append(read_bronze(fp))

    silver = pd.concat(dfs, ignore_index=True)
    silver.sort_values(["cik", "end"], inplace=True)

    if FILL_MISSING:
        # Forward‑fill fundamentals within each company to daily frequency.
        # Useful when you plan to join on price later.
        silver = (
            silver.set_index("end")
                  .groupby("cik").apply(lambda g: g.asfreq("D").ffill())
                  .reset_index(level=0, drop=True)
                  .reset_index()
        )

    silver.to_parquet(OUTPUT_FILE)
    try:
        rel = OUTPUT_FILE.resolve().relative_to(Path.cwd())
    except ValueError:
        rel = OUTPUT_FILE.resolve()
    print(f"✅  {rel} written ({len(silver)} rows, {silver.cik.nunique()} companies)")


if __name__ == "__main__":
    main(sys.argv[1:])
