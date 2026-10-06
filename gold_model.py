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

import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_config import (
    FEATURE_COLS,
    FEATURES_OUT,
    MODEL_OUT,
    PREPROCESSOR_OUT,
    SCALER_OUT,
    SKLEARN_MODEL_OUT,
)

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    KERAS_AVAILABLE = True
except ModuleNotFoundError:
    KERAS_AVAILABLE = False

# ── CONFIG ───────────────────────────────────────────────────────────
TARGET = "excess_ret"         # regression target (float)
EPOCHS = int(os.getenv("EPOCHS", "50"))
# TARGET = "label_up"         # alternative: classification (0/1)

# ── LOAD FEATURES ───────────────────────────────────────────────────
print("📥  Loading features …")
df = pd.read_parquet(FEATURES_OUT)

for col in FEATURE_COLS:
    if col not in df:
        df[col] = np.nan

# Replace +/-Inf with NaN; imputation handles sparse feature families.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=[TARGET]).copy()
df["date"] = pd.to_datetime(df["date"], utc=True)
df.sort_values("date", inplace=True)
print(f"Rows after cleaning: {len(df)}")

if df.empty:
    raise RuntimeError("No training rows with a non-null target.")

split_idx = max(int(len(df) * 0.8), 1)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]
if test_df.empty:
    test_df = train_df.copy()

preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
)
X_train = preprocessor.fit_transform(train_df[FEATURE_COLS]).astype(np.float32)
X_test = preprocessor.transform(test_df[FEATURE_COLS]).astype(np.float32)
y_train = train_df[TARGET].values.astype(np.float32)
y_test = test_df[TARGET].values.astype(np.float32)

assert np.isfinite(X_train).all() and np.isfinite(X_test).all(), "Non-finite values after preprocessing"
print(f"Train rows: {len(y_train)}, Test rows: {len(y_test)}")
print(f"Model input features: {len(FEATURE_COLS)}")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

if KERAS_AVAILABLE:
    # ── DEFINE MODEL ────────────────────────────────────────────────
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.20),
        Dense(64, activation="relu"),
        Dropout(0.10),
        Dense(1),                          # regression output
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    # ── TRAIN MODEL ─────────────────────────────────────────────────
    print("🚀  Training TensorFlow/Keras model …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
        verbose=2,
    )

    # ── EVALUATE MODEL ──────────────────────────────────────────────
    print("📊  Evaluating model …")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.5f}")

    model.save(MODEL_OUT)
    print(f"✅  Saved model to {MODEL_OUT}")
else:
    print("TensorFlow is not installed; training sklearn MLP fallback.")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        random_state=42,
        max_iter=max(EPOCHS, 1),
        batch_size=256,
        early_stopping=False,
        verbose=True,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.5f}")
    dump(model, SKLEARN_MODEL_OUT)
    print(f"✅  Saved sklearn fallback model to {SKLEARN_MODEL_OUT}")

# ── SAVE SCALER ─────────────────────────────────────────────────────
dump(preprocessor, PREPROCESSOR_OUT)
dump(preprocessor, SCALER_OUT)
print(f"✅  Saved preprocessor to {PREPROCESSOR_OUT}")
