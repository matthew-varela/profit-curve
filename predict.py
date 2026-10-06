# predict.py — generate 63-day excess-return predictions using the trained gold model
# ============================================================================
# Usage:
#   python predict.py                       # predict on the latest feature matrix
#   python predict.py --features path.parquet --out preds.parquet
#
# The script replicates the exact preprocessing used in gold_model.py:
#   • selects the shared 100+ feature list
#   • applies the saved imputer + StandardScaler pipeline
#   • feeds the tensor into the Keras model
#
# It writes a parquet (or prints a sample) with columns [date, ticker, pred_excess_ret].
# ============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from feature_config import FEATURE_COLS, FEATURES_OUT, MODEL_OUT, PREPROCESSOR_OUT, SKLEARN_MODEL_OUT

# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions with the gold model")
    p.add_argument("--features", type=Path, default=FEATURES_OUT, help="Input features parquet")
    p.add_argument("--out",      type=Path, default=Path("data/predictions.parquet"), help="Output predictions parquet")
    p.add_argument("--latest",   action="store_true",             help="Keep only the most-recent date per ticker in output")
    return p.parse_args()

# ── MAIN ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1) Load feature matrix
    print("📥  Loading features …")
    df = pd.read_parquet(args.features)

    # 2) Pre-processing identical to training
    for col in FEATURE_COLS:
        if col not in df:
            df[col] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean = df.copy()
    if clean.empty:
        raise RuntimeError("No rows available after loading features.")

    # 3) Load preprocessor and transform
    print("🔧  Applying saved preprocessor …")
    preprocessor = load(PREPROCESSOR_OUT)
    X = preprocessor.transform(clean[FEATURE_COLS]).astype(np.float32)

    # 4) Load model and predict
    print("🤖  Loading model …")
    if MODEL_OUT.exists():
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(MODEL_OUT)
            preds = model.predict(X, batch_size=32).flatten()
        except ModuleNotFoundError:
            if not SKLEARN_MODEL_OUT.exists():
                raise
            model = load(SKLEARN_MODEL_OUT)
            preds = model.predict(X).ravel()
    elif SKLEARN_MODEL_OUT.exists():
        model = load(SKLEARN_MODEL_OUT)
        preds = model.predict(X).ravel()
    else:
        raise FileNotFoundError(f"No trained model found at {MODEL_OUT} or {SKLEARN_MODEL_OUT}")

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