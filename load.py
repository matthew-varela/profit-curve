from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq        # faster row-count, optional

root = Path.cwd()                   # absolute project root

for p in Path("data").rglob("*.parquet"):
    rel  = p.resolve().relative_to(root)   # now both absolute
    rows = pq.ParquetFile(p).metadata.num_rows   # or len(pd.read_parquet(p))
    print(f"{rel} — {rows:,} rows")
