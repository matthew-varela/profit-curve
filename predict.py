# predict.py — generate 63-day excess-return predictions using the trained gold model
# ============================================================================
# Usage:
#   python predict.py                       # predict on the latest feature matrix
#   python predict.py --features path.parquet --out preds.parquet
#
# The script replicates the exact preprocessing used in gold_model.py:
#   • adds log1p(market_cap) column
#   • applies the saved StandardScaler
#   • feeds the tensor into the Keras model
#
# It writes a parquet (or prints a sample) with columns [date, ticker, pred_excess_ret].
# ============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load, dump
import tensorflow as tf

# ── DEFAULT PATHS ────────────────────────────────────────────────────
FEATURES_DEFAULT = Path("data/features.parquet")
MODEL_FILE       = Path("models/gold_model.keras")
SCALER_FILE      = Path("models/feature_scaler.pkl")
OUT_DEFAULT      = Path("data/predictions.parquet")

FEATURE_COLS = [
    "rev_qoq",
    "debt_equity",
    "gross_margin",
    "market_cap_log",  # added below
]

# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions with the gold model")
    p.add_argument("--features", type=Path, default=FEATURES_DEFAULT, help="Input features parquet (default data/features.parquet)")
    p.add_argument("--out",      type=Path, default=OUT_DEFAULT,    help="Output predictions parquet (default data/predictions.parquet)")
    p.add_argument("--latest",   action="store_true",             help="Keep only the most-recent date per ticker in output")
    return p.parse_args()

# ── MAIN ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1) Load feature matrix
    print("📥  Loading features …")
    df = pd.read_parquet(args.features)

    # 2) Pre-processing identical to training
    df["market_cap_log"] = np.log1p(df["market_cap"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean = df.dropna(subset=FEATURE_COLS)
    if clean.empty:
        raise RuntimeError("No rows with complete features after cleaning — cannot predict.")

    # 3) Load scaler and transform
    print("🔧  Applying scaler …")
    try:
        scaler = load(SCALER_FILE)
    except FileNotFoundError:
        print(f"⚠️  Scaler file {SCALER_FILE} not found — fitting a new scaler on the fly (results may differ from training).")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(clean[FEATURE_COLS].astype(np.float32))
        # persist so subsequent runs are consistent
        SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump(scaler, SCALER_FILE)
        print(f"✅  Saved new scaler to {SCALER_FILE}")
    X = scaler.transform(clean[FEATURE_COLS].astype(np.float32))

    # 4) Load model and predict
    print("🤖  Loading model …")
    model = tf.keras.models.load_model(MODEL_FILE)
    preds = model.predict(X, batch_size=32).flatten()

    clean = clean.copy()
    clean["pred_excess_ret"] = preds

    if args.latest:
        # Keep the most-recent date for each ticker
        clean.sort_values("date", inplace=True)
        clean = clean.groupby("ticker").tail(1)

    out_cols = ["date", "ticker", "pred_excess_ret"]
    out_df = clean[out_cols].reset_index(drop=True)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out)
    print(f"✅  Wrote {len(out_df):,} predictions → {args.out}")
    print(out_df.head())


if __name__ == "__main__":
    main() 