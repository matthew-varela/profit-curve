# gold_model.py — train model on feature matrix to predict 63‑day excess return
# ============================================================================
# 1. Loads the daily feature matrix built by feature_build.py.
# 2. Cleans out any NaN/Inf rows so Keras never sees non‑finite numbers.
# 3. Trains a simple 2‑layer MLP to regress 63‑day excess return.
# 4. Saves the model to models/gold_model.keras
#
# You can switch TARGET to "label_up" and change the last Dense layer + loss
# if you prefer a binary‑classification framing.
# ============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from joblib import dump

# ── CONFIG ───────────────────────────────────────────────────────────
FEATURES_IN = Path("data/features.parquet")
MODEL_OUT   = Path("models/gold_model.keras")
SCALER_OUT  = Path("models/feature_scaler.pkl")

# ── FEATURE SELECTION ────────────────────────────────────────────────
FEATURE_COLS = [
    "rev_qoq",
    "debt_equity",
    "gross_margin",
    "market_cap_log",  # log-scaled version added below
]
TARGET = "excess_ret"         # regression target (float)
# TARGET = "label_up"         # alternative: classification (0/1)

# ── LOAD FEATURES ───────────────────────────────────────────────────
print("📥  Loading features …")
df = pd.read_parquet(FEATURES_IN)

# Replace raw market_cap with log-1p to keep magnitudes reasonable (≈ 6-13)
df["market_cap_log"] = np.log1p(df["market_cap"])

# Replace ±Inf with NaN, then drop any remaining NaNs in required columns
finite_cols = FEATURE_COLS + [TARGET]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
print(f"Rows after cleaning: {len(df)}")

# Assert everything is finite before training
assert np.isfinite(df[finite_cols].values).all(), "Non‑finite values still present after cleaning"

# Standardize to zero-mean / unit-var — helps optimizer avoid NaNs
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURE_COLS]).astype(np.float32)
# Optionally you might persist the scaler with joblib for inference
y = df[TARGET].values.astype(np.float32)

# ── TRAIN / TEST SPLIT ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train rows: {len(y_train)}, Test rows: {len(y_test)}")

# ── DEFINE MODEL ────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1),                          # regression output
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

# ── TRAIN MODEL ─────────────────────────────────────────────────────
print("🚀  Training model …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=2,
)

# ── EVALUATE MODEL ──────────────────────────────────────────────────
print("📊  Evaluating model …")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.5f}")

# ── SAVE MODEL ──────────────────────────────────────────────────────
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_OUT)
print(f"✅  Saved model to {MODEL_OUT}")

# ── SAVE SCALER ─────────────────────────────────────────────────────
dump(scaler, SCALER_OUT)
print(f"✅  Saved scaler to {SCALER_OUT}")
