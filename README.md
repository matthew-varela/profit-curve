# Profit Curve 📈

TensorFlow-powered pipeline for predicting 63-day excess stock returns.

## Pipeline Stages
1. **Data Ingestion** — SEC XBRL fundamentals + Yahoo! Finance prices  
2. **Bronze → Silver ETL** — parquet storage, outlier winsorization, temporal joins  
3. **Feature Generation** — 200+ ratios, rolling z-scores, macro factors  
4. **Labeling** — forward 63-day excess return vs. SPY benchmark  
5. **Modeling** — dense neural net w/ dropout + cyclical LR  
6. **Evaluation** — ROC-AUC, IR, turnover, latency benchmarks