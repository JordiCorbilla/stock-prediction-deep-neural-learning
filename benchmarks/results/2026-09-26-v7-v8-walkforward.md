# v7 vs v8 rigorous walk-forward benchmark — 2026-09-26

This report records the deterministic cross-asset benchmark executed by GitHub Actions on branch `architecture-clean-up`.

Workflow run: https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/actions/runs/36274776510

**Reproduction limit:** The committed fold-skill CSV supports the comparative skill counts below. Raw market and prediction snapshots were stored as workflow artifacts rather than committed here; their bytes are not available from this report. Exact RMSE, correlation, and strategy statistics therefore require the original artifacts and should not be treated as independently reproduced from this checkout. The original BTC-USD strategy diagnostics used a 252-period annualisation; new benchmark runs use 365 for that continuous market. Neither model is promoted as a superior forecaster because aggregate naive-relative skill is negative.

## Protocol

- Data source: Yahoo Finance via `yfinance`.
- Prices: adjusted close (`auto_adjust=True`).
- History start: 2015-01-01.
- Test regimes: 2020, 2022, 2024, 2026 YTD.
- Validation: the calendar year immediately before each test regime.
- Training: all observations before the validation year.
- Forecast: rolling one-step-ahead; each prediction may use observed history available before that target date.
- Input window: 30 observations.
- Maximum epochs: 20 with early stopping.
- Batch size: 64.
- Training order: chronological (`shuffle=False`).
- Seed: 42.
- Transaction-cost diagnostic: 5 bps.
- Baseline: previous close, i.e. `P[t] = P[t-1]`.
- Assets: SPY, QQQ, AAPL, FTSE 100, BTC-USD.

The input scaler and target-magnitude scaler are fitted only on the training portion of each fold. Validation data controls early stopping. Test data does not select model weights or hyperparameters.

## Aggregate results

| Asset | Model | Observations | Folds | RMSE skill vs naive | Directional accuracy | Delta IC | Positive-skill folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | v7 | 940 | 4 | -0.1990 | 0.5266 | -0.0232 | 0 |
| AAPL | v8 | 940 | 4 | **-0.0591** | 0.5106 | **0.0251** | 0 |
| BTC-USD | v7 | 1366 | 4 | -0.2106 | 0.4971 | -0.0110 | 0 |
| BTC-USD | v8 | 1366 | 4 | **-0.0670** | **0.5051** | **-0.0027** | 1 |
| FTSE 100 | v7 | 943 | 4 | **-0.1263** | 0.5281 | **0.0584** | 0 |
| FTSE 100 | v8 | 943 | 4 | -0.1382 | 0.5281 | 0.0464 | 0 |
| QQQ | v7 | 940 | 4 | -0.1636 | 0.5500 | 0.0201 | 0 |
| QQQ | v8 | 940 | 4 | **-0.0645** | 0.5500 | **0.0279** | 1 |
| SPY | v7 | 940 | 4 | -0.1789 | 0.5298 | 0.0198 | 0 |
| SPY | v8 | 940 | 4 | **-0.0884** | 0.5298 | **0.0283** | 1 |

A positive RMSE skill means a model beat the previous-close naive forecast. Negative skill means the naive forecast had lower RMSE.

Across the five asset-level aggregates, v8 has better RMSE skill on **4/5 assets**. Across all 20 asset/regime folds, v8 has better RMSE skill than v7 on **17/20 folds**.

Mean fold RMSE skill:

- v7: **-0.1937**
- v8: **-0.0744**

Median fold RMSE skill:

- v7: **-0.1479**
- v8: **-0.0542**

Positive RMSE-skill folds:

- v7: **0/20**
- v8: **3/20**

Mean directional accuracy across folds:

- v7: **52.62%**
- v8: **52.47%**

Mean delta information coefficient across folds:

- v7: **-0.0344**
- v8: **0.0174**

## Fold-level RMSE skill

| Asset | Test year | v7 skill | v8 skill | Better architecture |
| --- | ---: | ---: | ---: | --- |
| AAPL | 2020 | -0.0496 | **-0.0033** | v8 |
| AAPL | 2022 | -0.3222 | **-0.0297** | v8 |
| AAPL | 2024 | -0.3033 | **-0.0342** | v8 |
| AAPL | 2026 | -0.1137 | **-0.1075** | v8 |
| QQQ | 2020 | -0.0631 | **0.0036** | v8 |
| QQQ | 2022 | -0.1515 | **-0.0546** | v8 |
| QQQ | 2024 | -0.2962 | **-0.0774** | v8 |
| QQQ | 2026 | -0.1423 | **-0.0907** | v8 |
| FTSE 100 | 2020 | **-0.0847** | -0.1656 | v7 |
| FTSE 100 | 2022 | **-0.1416** | -0.1471 | v7 |
| FTSE 100 | 2024 | -0.3130 | **-0.1965** | v8 |
| FTSE 100 | 2026 | -0.1151 | **-0.0240** | v8 |
| BTC-USD | 2020 | -0.2783 | **0.0099** | v8 |
| BTC-USD | 2022 | -0.3002 | **-0.0316** | v8 |
| BTC-USD | 2024 | -0.1313 | **-0.0172** | v8 |
| BTC-USD | 2026 | -0.2786 | **-0.1586** | v8 |
| SPY | 2020 | -0.0546 | **0.0012** | v8 |
| SPY | 2022 | -0.1443 | **-0.0538** | v8 |
| SPY | 2024 | -0.3825 | **-0.0835** | v8 |
| SPY | 2026 | **-0.2070** | -0.2280 | v7 |

The three positive v8 folds are QQQ 2020, BTC-USD 2020 and SPY 2020. These gains are small and should not be interpreted as a persistent forecasting edge.

## Interpretation

The benchmark supports two separate conclusions.

First, the shared multi-task v8 architecture is a meaningful architectural improvement over v7. It produces lower RMSE relative to the naive baseline in 17 of 20 folds and wins four of the five asset-level aggregates. Its average delta IC is also modestly positive while v7's is negative.

Second, neither architecture currently demonstrates robust price-forecasting skill beyond the previous-close baseline. v7 never beats the naive baseline in any of the 20 folds. v8 beats it in only three folds, all in 2020, and the improvements are very small.

This is precisely why the repository now treats the naive baseline as mandatory: visually plausible predicted-price charts can coexist with negative forecast skill.

## Trading-diagnostic caveat

Directional strategy metrics are retained in the raw benchmark artifacts but are secondary here. Several folds collapse to an almost constant directional position, which makes the resulting strategy resemble long-only exposure rather than a useful switching signal. Forecast RMSE skill and delta IC are therefore the primary metrics for this architecture comparison.

## Reproducibility artifacts

The workflow retains, per asset:

- frozen adjusted-close market-data CSV;
- fold-level metrics CSV;
- every held-out prediction;
- JSON summary.

The aggregate workflow also retains `aggregate.csv` and `BENCHMARK_SUMMARY.md`.

The benchmark harness is:

`benchmarks/run_v7_v8_walkforward.py`

The execution workflow is:

`.github/workflows/research-benchmark.yml`

## Next research implication

v8 should remain an experimental model rather than replacing v7 as the compatibility default. If further model research is pursued, the next benchmark should target returns/deltas directly and compare against a modern time-series architecture such as PatchTST or N-HiTS using the same frozen walk-forward protocol.
