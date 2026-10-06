"""Build a 100+ column feature matrix with fundamentals, OHLCV, and filing NLP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_config import (
    BENCHMARK_TICKER,
    FEATURE_COLS,
    FEATURES_OUT,
    FILING_FEATURES_OUT,
    FUNDAMENTAL_BASES,
    FUNDAMENTAL_LAG_WINDOWS,
    INFO_LAG_DAYS,
    LABEL_HORIZON_DAYS,
    MAX_FF_DAYS,
    NLP_EMBEDDING_FEATURES,
    NLP_STAT_FEATURES,
    OHLCV_WINDOWS,
    PRICE_DIR,
    SILVER_DIR,
    TICKER_CIK,
)

SILVER_FUND = SILVER_DIR / "fundamentals.parquet"
SHARES_CSV = PRICE_DIR / "shares.csv"
EPS = 1e-9


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join(str(c) for c in tup if c) for tup in df.columns]
    return df


def pick_column(df: pd.DataFrame, candidates: list[str], fallback: pd.Series | None = None) -> pd.Series:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        for low, original in lower.items():
            if candidate in low:
                return pd.to_numeric(df[original], errors="coerce")
    if fallback is not None:
        return fallback
    return pd.Series(np.nan, index=df.index, dtype=float)


def load_price_frame(ticker: str) -> pd.DataFrame:
    raw = flatten_columns(pd.read_parquet(PRICE_DIR / f"{ticker}.parquet"))
    raw.index = pd.to_datetime(raw.index, utc=True)
    close = pick_column(raw, ["adj close", "adj_close", "close"])
    frame = pd.DataFrame(index=raw.index)
    frame["open"] = pick_column(raw, ["open"], close)
    frame["high"] = pick_column(raw, ["high"], close)
    frame["low"] = pick_column(raw, ["low"], close)
    frame["close"] = pick_column(raw, ["close"], close)
    frame["adj_close"] = close
    frame["volume"] = pick_column(raw, ["volume"], pd.Series(0.0, index=raw.index))
    return frame.sort_index().asfreq("D").ffill()


def rolling_slope(values: np.ndarray) -> float:
    if np.isnan(values).any() or np.nanstd(values) == 0:
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def rolling_r2(values: np.ndarray) -> float:
    if np.isnan(values).any() or np.nanstd(values) == 0:
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((values - pred) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    return np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)


def add_ohlcv_features(price: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    out = price.copy()
    out["ret_1d"] = out["adj_close"].pct_change(fill_method=None)
    out["log_ret_1d"] = np.log(out["adj_close"]).diff()
    out["range_1d"] = (out["high"] - out["low"]) / (out["close"].abs() + EPS)
    out["gap_1d"] = (out["open"] / (out["close"].shift(1) + EPS)) - 1
    out["close_pos_1d"] = (out["close"] - out["low"]) / ((out["high"] - out["low"]) + EPS)
    out["dollar_volume"] = out["adj_close"] * out["volume"]
    spy_ret_1d = spy_close.pct_change(fill_method=None).reindex(out.index).ffill()

    for window in OHLCV_WINDOWS:
        out[f"ret_{window}d"] = out["adj_close"].pct_change(window, fill_method=None)
        out[f"log_ret_{window}d"] = np.log(out["adj_close"]).diff(window)
        out[f"vol_{window}d"] = out["ret_1d"].rolling(window).std() * np.sqrt(252)
        out[f"downside_vol_{window}d"] = (
            out["ret_1d"].clip(upper=0).rolling(window).std() * np.sqrt(252)
        )
        out[f"range_mean_{window}d"] = out["range_1d"].rolling(window).mean()
        out[f"gap_mean_{window}d"] = out["gap_1d"].rolling(window).mean()
        out[f"close_pos_mean_{window}d"] = out["close_pos_1d"].rolling(window).mean()
        volume_mean = out["volume"].rolling(window).mean()
        volume_std = out["volume"].rolling(window).std()
        out[f"volume_z_{window}d"] = (out["volume"] - volume_mean) / (volume_std + EPS)
        out[f"volume_ratio_{window}d"] = (out["volume"] / (volume_mean + EPS)) - 1
        out[f"dollar_volume_log_{window}d"] = np.log1p(out["dollar_volume"].rolling(window).mean())
        out[f"drawdown_{window}d"] = (out["adj_close"] / (out["adj_close"].rolling(window).max() + EPS)) - 1
        log_price = np.log(out["adj_close"])
        out[f"trend_slope_{window}d"] = log_price.rolling(window).apply(rolling_slope, raw=True)
        out[f"trend_r2_{window}d"] = log_price.rolling(window).apply(rolling_r2, raw=True)
        out[f"ret_skew_{window}d"] = out["ret_1d"].rolling(window).skew()
        out[f"ret_kurt_{window}d"] = out["ret_1d"].rolling(window).kurt()
        out[f"ret_autocorr_{window}d"] = out["ret_1d"].rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=1),
            raw=False,
        )
        spy_ret_window = spy_close.pct_change(window, fill_method=None).reindex(out.index).ffill()
        out[f"spy_beta_{window}d"] = out["ret_1d"].rolling(window).cov(spy_ret_1d) / (
            spy_ret_1d.rolling(window).var() + EPS
        )
        out[f"spy_corr_{window}d"] = out["ret_1d"].rolling(window).corr(spy_ret_1d)
        out[f"excess_ret_{window}d"] = out[f"ret_{window}d"] - spy_ret_window
        out[f"volume_ret_corr_{window}d"] = out["volume"].pct_change(fill_method=None).rolling(window).corr(out["ret_1d"])

    return out


def load_fundamentals_daily() -> pd.DataFrame:
    fund = pd.read_parquet(SILVER_FUND)
    fund["cik"] = fund["cik"].astype(str).str.zfill(10)
    fund["end"] = pd.to_datetime(fund["end"], utc=True)
    fund["info_date"] = fund["end"] + pd.Timedelta(days=INFO_LAG_DAYS)

    parts = []
    for cik, g in fund.groupby("cik"):
        g = g.sort_values(["info_date", "end"]).drop_duplicates("info_date", keep="last")
        daily = (
            g.set_index("info_date")
            .sort_index()
            .asfreq("D")
            .ffill(limit=MAX_FF_DAYS)
            .assign(cik=cik)
        )
        parts.append(daily)
    return pd.concat(parts).reset_index(names="date")


def add_fundamental_features(features: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "assets",
        "liabilities",
        "equity",
        "revenue",
        "cogs",
        "net_income",
        "operating_cf",
        "capex",
        "eps_diluted",
        "cash",
        "current_assets",
        "current_liabilities",
        "intangibles",
        "goodwill",
    ]:
        if col not in features:
            features[col] = np.nan

    features["market_cap_log"] = np.log1p(features["market_cap"])
    features["debt_equity"] = features["liabilities"] / (features["equity"] + EPS)
    features["liabilities_assets"] = features["liabilities"] / (features["assets"] + EPS)
    features["equity_assets"] = features["equity"] / (features["assets"] + EPS)
    features["gross_margin"] = 1 - (features["cogs"] / (features["revenue"] + EPS))
    features["net_margin"] = features["net_income"] / (features["revenue"] + EPS)
    features["operating_cf_margin"] = features["operating_cf"] / (features["revenue"] + EPS)
    features["capex_revenue"] = features["capex"] / (features["revenue"] + EPS)
    features["fcf_margin"] = (features["operating_cf"] - features["capex"]) / (features["revenue"] + EPS)
    features["accruals_assets"] = (features["net_income"] - features["operating_cf"]) / (features["assets"] + EPS)
    features["asset_turnover"] = features["revenue"] / (features["assets"] + EPS)
    features["roe"] = features["net_income"] / (features["equity"] + EPS)
    features["roa"] = features["net_income"] / (features["assets"] + EPS)
    features["revenue_per_share"] = features["revenue"] / (features["shares_outstanding"] + EPS)
    features["eps_to_price"] = features["eps_diluted"] / (features["adj_close"] + EPS)
    features["sales_to_market_cap"] = features["revenue"] / (features["market_cap"] + EPS)
    features["cash_assets"] = features["cash"] / (features["assets"] + EPS)
    features["current_ratio"] = features["current_assets"] / (features["current_liabilities"] + EPS)
    features["working_capital_assets"] = (
        features["current_assets"] - features["current_liabilities"]
    ) / (features["assets"] + EPS)
    features["intangibles_assets"] = features["intangibles"] / (features["assets"] + EPS)
    features["goodwill_assets"] = features["goodwill"] / (features["assets"] + EPS)

    for base in FUNDAMENTAL_BASES:
        if base not in features:
            features[base] = np.nan
        grouped = features.groupby("ticker")[base]
        for window in FUNDAMENTAL_LAG_WINDOWS:
            features[f"{base}_pct_change_{window}d"] = grouped.transform(
                lambda x, w=window: x.pct_change(w, fill_method=None)
            )
            features[f"{base}_stability_{window}d"] = grouped.transform(
                lambda x, w=window: x.rolling(w).std() / (x.rolling(w).mean().abs() + EPS)
            )
    return features


def load_shares() -> dict[str, float]:
    shares = pd.read_csv(SHARES_CSV)
    shares["shares_outstanding"] = pd.to_numeric(shares["shares_outstanding"], errors="coerce")
    return shares.set_index("ticker")["shares_outstanding"].to_dict()


def merge_filing_features(features: pd.DataFrame) -> pd.DataFrame:
    if not FILING_FEATURES_OUT.exists():
        for col in NLP_STAT_FEATURES + NLP_EMBEDDING_FEATURES:
            features[col] = np.nan
        return features

    filing = pd.read_parquet(FILING_FEATURES_OUT)
    filing["cik"] = filing["cik"].astype(str).str.zfill(10)
    filing["filing_date"] = pd.to_datetime(filing["filing_date"], utc=True)
    feature_cols = NLP_STAT_FEATURES + NLP_EMBEDDING_FEATURES
    parts = []
    for _, g in features.groupby("ticker", sort=False):
        cik = str(g["cik"].dropna().iloc[0]).zfill(10) if g["cik"].notna().any() else ""
        filing_g = filing[filing["cik"] == cik][["filing_date"] + feature_cols].sort_values("filing_date")
        if filing_g.empty:
            for col in feature_cols:
                g[col] = np.nan
            parts.append(g)
            continue
        merged = pd.merge_asof(
            g.sort_values("date"),
            filing_g,
            left_on="date",
            right_on="filing_date",
            direction="backward",
        ).drop(columns=["filing_date"])
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    print("Loading fundamentals...")
    fund_daily = load_fundamentals_daily()
    shares_map = load_shares()

    print("Loading benchmark...")
    spy = load_price_frame(BENCHMARK_TICKER)

    print("Building ticker features...")
    parts = []
    for tkr, cik in TICKER_CIK.items():
        price = add_ohlcv_features(load_price_frame(tkr), spy["adj_close"])
        price = price.reset_index(names="date")
        f = fund_daily[fund_daily["cik"].str.lstrip("0") == cik.lstrip("0")]
        merged = price.merge(f, on="date", how="left")
        shares = shares_map.get(tkr)
        merged["shares_outstanding"] = shares
        merged["market_cap"] = merged["adj_close"] * shares if pd.notna(shares) else np.nan
        merged["ticker"] = tkr
        merged["cik"] = cik
        parts.append(merged)

    features = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
    features = add_fundamental_features(features)
    features = merge_filing_features(features)

    features = features.merge(
        spy["adj_close"].rename("spy_close"),
        left_on="date",
        right_index=True,
        how="left",
    )
    features["fwd_adj_close"] = features.groupby("ticker")["adj_close"].shift(-LABEL_HORIZON_DAYS)
    features["fwd_spy_close"] = features.groupby("ticker")["spy_close"].shift(-LABEL_HORIZON_DAYS)
    features["future_return"] = (features["fwd_adj_close"] / features["adj_close"]) - 1
    features["spy_future_ret"] = (features["fwd_spy_close"] / features["spy_close"]) - 1
    features["excess_ret"] = features["future_return"] - features["spy_future_ret"]
    features["label_up"] = (features["excess_ret"] > 0).astype(int)
    features.drop(columns=["fwd_adj_close", "fwd_spy_close"], inplace=True)

    for col in FEATURE_COLS:
        if col not in features:
            features[col] = np.nan

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_OUT)
    print(
        f"Saved {FEATURES_OUT}: {len(features):,} rows, "
        f"{features.ticker.nunique()} tickers, {len(FEATURE_COLS):,} configured model features"
    )


if __name__ == "__main__":
    main()
