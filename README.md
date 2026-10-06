# Profit Curve 📈

End-to-end machine learning pipeline that forecasts **63-trading-day forward excess returns vs. SPY** for a universe of 30 large-cap U.S. equities. It fuses SEC XBRL fundamentals, SEC 10-K/10-Q filing text, and daily Yahoo! Finance OHLCV data into a point-in-time feature matrix, then trains a TensorFlow/Keras neural network on it.

| | |
|---|---|
| Universe | 30 large caps (AAPL, MSFT, NVDA, AMZN, GOOGL, META, BRK-B, JPM, XOM, LLY, …) + SPY benchmark |
| History | Daily, 2010-01-01 → present |
| Feature matrix | ~177K ticker-day rows × 288 model features |
| Target | `excess_ret` = 63-day forward return − SPY 63-day forward return |
| Model | Keras MLP (128 → 64, ReLU, dropout), sklearn `MLPRegressor` fallback |

## Pipeline

Run the stages in order from the project root:

```bash
python price_download.py     # 1. OHLCV + shares outstanding (yfinance)
python sec_download.py       # 2. SEC companyfacts JSON + recent 10-K/10-Q text
python bronze_clean.py       # 3. Raw XBRL JSON → per-company Parquet (bronze)
python silver_join.py        # 4. Merge bronze tables → data/silver/fundamentals.parquet
python filing_features.py    # 5. Filing text → NLP stats + embeddings
python feature_build.py      # 6. Build 288-feature matrix + labels
python gold_model.py         # 7. Train model (EPOCHS env var, default 50)
python predict.py --latest   # 8. Write predictions to data/predictions.parquet
```

Most scripts accept an optional ticker/CIK subset, e.g. `python sec_download.py AAPL MSFT --filing-limit 4`.

### 1–2. Data ingestion
- **Prices:** split/dividend-adjusted daily OHLCV for each ticker and SPY, plus shares outstanding.
- **Fundamentals:** SEC EDGAR XBRL `companyfacts` API.
- **Filings:** the most recent 10-K/10-Q primary documents per company via the EDGAR `submissions` and Archives endpoints, converted from HTML to plain text with BeautifulSoup.
- Requests use an SEC-compliant `User-Agent`, are throttled to the SEC's 10 req/s limit, and retry with exponential backoff + jitter. Tickers resolve to CIKs through the configured map, falling back to yfinance metadata and SEC ticker files.

### 3–4. Bronze → Silver ETL
- Maps 40+ US-GAAP XBRL tags (e.g. four alternative stockholders'-equity tags) to 22 canonical fields such as `revenue`, `net_income`, `operating_cf`, `equity`, and `long_term_debt`.
- Keeps Q1–Q4/FY periods, de-duplicates restatements (latest value wins), and pivots long → wide.
- Concatenates all companies into a single silver fundamentals table.

### 5. Filing NLP features (73)
- **Text statistics (9):** length, word/sentence counts, average word length, numeric density, and dictionary-based uncertainty, risk, and litigation term rates.
- **Embeddings (64):** `sentence-transformers/all-MiniLM-L6-v2` embeddings reduced to 64 dimensions with Gaussian random projection. Falls back to a 512-feature `HashingVectorizer` if sentence-transformers is unavailable.
- **Disclosure change:** cosine similarity of each filing to the same company's previous filing.

### 6. Feature engineering (288 total)
- **OHLCV (140):** 20 feature families × 7 windows (5, 10, 21, 42, 63, 126, 252 days). Includes returns, annualized and downside volatility, drawdown, skew, kurtosis, autocorrelation, rolling SPY beta/correlation, excess momentum, log-price trend slope and R², volume z-scores, dollar volume, gaps, and intraday range.
- **Fundamental ratios (21):** margins, ROE/ROA, accruals, leverage, liquidity, asset turnover, earnings yield, sales/market cap, intangibles and goodwill intensity, and log market cap.
- **Fundamental dynamics (54):** percent change and coefficient of variation over 63/126/252 days for 9 base metrics.

**Point-in-time handling:**
- Fundamentals become available 45 days after period end (`INFO_LAG_DAYS`) and are forward-filled for at most 400 days.
- Filing features are attached with a backward `merge_asof` on filing date.

### 7–8. Modeling & inference
- Rows are sorted by date and split chronologically 80/20 into train and test.
- A scikit-learn `Pipeline` (median `SimpleImputer` → `StandardScaler`) is fit on the training split only.
- **Keras model:** `Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dropout(0.1) → Dense(1)`, trained with Adam and MSE loss, reporting test MAE.
- If TensorFlow isn't installed, an sklearn `MLPRegressor` with the same layer sizes is trained instead.
- `predict.py` reloads the saved preprocessor and model and writes `[date, ticker, pred_excess_ret]` to Parquet.

## Configuration

All shared settings live in `feature_config.py`: the ticker → CIK universe, start date, benchmark, label horizon, information lag, rolling windows, embedding size, and file paths.

## Project layout

```
feature_config.py     shared configuration
price_download.py     yfinance OHLCV + shares outstanding
sec_download.py       SEC companyfacts + filing text
bronze_clean.py       XBRL JSON → bronze Parquet
silver_join.py        bronze → silver fundamentals
filing_features.py    filing NLP features
feature_build.py      feature matrix + labels
gold_model.py         model training
predict.py            inference
load.py               Parquet row-count utility
data/                 raw/, bronze/, silver/, filings/, price Parquet, outputs
models/               trained model + fitted preprocessor
```

`data/features.parquet` is generated by `feature_build.py` and is not tracked in git (it exceeds GitHub's file-size limit).

## Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins TensorFlow 2.16 with `tensorflow-macos`/`tensorflow-metal` for Apple Silicon GPU acceleration; on other platforms, drop those two lines.

## Known limitations

- **Static shares outstanding:** today's share count is applied to all of history, which introduces mild look-ahead bias into market cap and valuation ratios.
- **Overlapping labels:** consecutive 63-day labels overlap, and the train/test split has no purge or embargo gap, so test metrics are likely optimistic.
- **Survivorship bias:** the universe consists of today's large caps.
- **Shallow filing history:** only the most recent filings per company are downloaded (8 by default), so NLP features are sparse in early years.
