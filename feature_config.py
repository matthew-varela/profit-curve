"""Shared configuration for the expanded market feature pipeline."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data")
PRICE_DIR = DATA_DIR
RAW_DIR = DATA_DIR / "raw"
FILINGS_DIR = DATA_DIR / "filings"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
FEATURES_OUT = DATA_DIR / "features.parquet"
FILING_FEATURES_OUT = DATA_DIR / "filing_features.parquet"

MODELS_DIR = Path("models")
MODEL_OUT = MODELS_DIR / "gold_model.keras"
SKLEARN_MODEL_OUT = MODELS_DIR / "gold_model.joblib"
PREPROCESSOR_OUT = MODELS_DIR / "feature_preprocessor.pkl"
SCALER_OUT = MODELS_DIR / "feature_scaler.pkl"

START_DATE = "2010-01-01"
BENCHMARK_TICKER = "SPY"
INFO_LAG_DAYS = 45
MAX_FF_DAYS = 400
LABEL_HORIZON_DAYS = 63
NLP_EMBEDDING_DIMS = 64

# Reproducible, expanded large-cap sample. Keep this small enough for personal
# experimentation while still giving the 100+ feature model more cross-section.
TICKER_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "BRK-B": "0001067983",
    "LLY": "0000059478",
    "AVGO": "0001730168",
    "JPM": "0000019617",
    "XOM": "0000034088",
    "UNH": "0000731766",
    "V": "0001403161",
    "PG": "0000080424",
    "MA": "0001141391",
    "COST": "0000909832",
    "JNJ": "0000200406",
    "HD": "0000354950",
    "MRK": "0000310158",
    "ABBV": "0001551152",
    "WMT": "0000104169",
    "KO": "0000021344",
    "BAC": "0000070858",
    "PEP": "0000077476",
    "CRM": "0001108524",
    "ADBE": "0000796343",
    "NFLX": "0001065280",
    "AMD": "0000002488",
    "CSCO": "0000858877",
    "INTC": "0000050863",
}

OHLCV_WINDOWS = [5, 10, 21, 42, 63, 126, 252]
OHLCV_FEATURE_TEMPLATES = [
    "ret_{w}d",
    "log_ret_{w}d",
    "vol_{w}d",
    "downside_vol_{w}d",
    "range_mean_{w}d",
    "gap_mean_{w}d",
    "close_pos_mean_{w}d",
    "volume_z_{w}d",
    "volume_ratio_{w}d",
    "dollar_volume_log_{w}d",
    "drawdown_{w}d",
    "trend_slope_{w}d",
    "trend_r2_{w}d",
    "ret_skew_{w}d",
    "ret_kurt_{w}d",
    "ret_autocorr_{w}d",
    "spy_beta_{w}d",
    "spy_corr_{w}d",
    "excess_ret_{w}d",
    "volume_ret_corr_{w}d",
]

FUNDAMENTAL_FEATURES = [
    "market_cap_log",
    "debt_equity",
    "liabilities_assets",
    "equity_assets",
    "gross_margin",
    "net_margin",
    "operating_cf_margin",
    "capex_revenue",
    "fcf_margin",
    "accruals_assets",
    "asset_turnover",
    "roe",
    "roa",
    "revenue_per_share",
    "eps_to_price",
    "sales_to_market_cap",
    "cash_assets",
    "current_ratio",
    "working_capital_assets",
    "intangibles_assets",
    "goodwill_assets",
]

FUNDAMENTAL_BASES = [
    "revenue",
    "net_income",
    "operating_cf",
    "assets",
    "liabilities",
    "equity",
    "gross_margin",
    "debt_equity",
    "market_cap",
]
FUNDAMENTAL_LAG_WINDOWS = [63, 126, 252]
FUNDAMENTAL_LAG_FEATURES = [
    f"{base}_{kind}_{window}d"
    for base in FUNDAMENTAL_BASES
    for window in FUNDAMENTAL_LAG_WINDOWS
    for kind in ("pct_change", "stability")
]

NLP_STAT_FEATURES = [
    "filing_text_len_log",
    "filing_word_count_log",
    "filing_avg_word_len",
    "filing_sentence_count_log",
    "filing_uncertainty_rate",
    "filing_risk_rate",
    "filing_litigation_rate",
    "filing_numeric_rate",
    "filing_similarity_prev",
]
NLP_EMBEDDING_FEATURES = [f"filing_embed_{i:02d}" for i in range(NLP_EMBEDDING_DIMS)]


def ohlcv_feature_names() -> list[str]:
    return [
        template.format(w=window)
        for window in OHLCV_WINDOWS
        for template in OHLCV_FEATURE_TEMPLATES
    ]


FEATURE_COLS = (
    FUNDAMENTAL_FEATURES
    + FUNDAMENTAL_LAG_FEATURES
    + ohlcv_feature_names()
    + NLP_STAT_FEATURES
    + NLP_EMBEDDING_FEATURES
)


def available_tickers(include_benchmark: bool = False) -> list[str]:
    tickers = list(TICKER_CIK)
    if include_benchmark:
        return tickers + [BENCHMARK_TICKER]
    return tickers
