# Quant Forecast Lab

**Reproducible deep-learning research for financial time-series forecasting.**

[![CI](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18.1](https://img.shields.io/badge/TensorFlow-2.18.1-orange.svg)](https://www.tensorflow.org/)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

This repository began as an LSTM stock-price forecasting experiment in 2020. The current architecture keeps those historical entry points compatible while evolving the project into a more rigorous quantitative forecasting laboratory.

> The objective is not to make a chart that visually tracks a price series. The objective is to measure whether a model has **out-of-sample forecasting skill beyond simple baselines**, whether uncertainty is calibrated, and what remains after realistic trading frictions.

## Highlights

- TensorFlow LSTM models for price, return, delta and trend-residual targets.
- v7 direction + magnitude forecasting with recursive future inference.
- v8 shared multi-task LSTM architecture for direction and magnitude.
- Recorded experiment seeds and runtime metadata; exact numeric reproducibility can vary by hardware and backend.
- Training/validation/test separation for model fitting and evaluation.
- Naive last-price benchmark and naive-relative RMSE skill.
- Directional accuracy.
- Expanding-window walk-forward split primitives.
- One-step held-out residual bands, explicitly labelled as unvalidated for recursive horizons.
- Transaction-cost-aware directional strategy diagnostics.
- CSV benchmark CLI.
- CI, tests, packaging metadata and contributor templates.

## Why the benchmark matters

Financial price levels are persistent. A forecast can look excellent on a chart while failing to beat:

~~~text
tomorrow's price = today's price
~~~

For that reason, model error should not be presented in isolation. The maintained evaluation layer reports:

~~~text
RMSE skill = 1 - RMSE(model) / RMSE(naive)
~~~

A positive value means the model beats the last-observation baseline on RMSE for the evaluated sample.

## Quick start

### Conda

~~~bash
conda env create -f environment.yml
conda activate stock-prediction
~~~

### pip

~~~bash
python -m venv .venv
# activate the environment for your platform
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

Run the `stock_prediction_*.py` compatibility scripts and notebooks from a source checkout. The published package installs the `quant-forecast` CLI and library; it does not bundle those root scripts or historical notebooks.

For development:

~~~bash
python -m pip install -e ".[dev]" --no-deps
python -m pip install -r requirements.txt
pytest
ruff check quant_forecast_lab tests tests_runtime
~~~

## Train the compatibility model

The historical CLI remains supported:

~~~bash
python stock_prediction_deep_learning.py \
  -ticker=GOOG \
  -start_date=2017-11-01 \
  -validation_date=2022-09-01 \
  -epochs=150 \
  -batch_size=32 \
  -time_steps=30 \
  -use_returns=false \
  -model_version=v7 \
  -forecast_horizon=1 \
  -seed=42
~~~

The cleanup keeps v7 as the compatibility default so existing users do not silently receive different model semantics.

## Train the shared multi-task model

v8 is an opt-in model that shares one temporal representation and predicts both move direction and magnitude:

~~~text
                         +--> P(up)
input --> LSTM --> LSTM -|
                         +--> magnitude
~~~

Run it with:

~~~bash
python stock_prediction_deep_learning.py \
  -ticker=GOOG \
  -start_date=2017-11-01 \
  -validation_date=2022-09-01 \
  -epochs=150 \
  -batch_size=32 \
  -time_steps=30 \
  -use_returns=false \
  -model_version=v8 \
  -seed=42
~~~

v7 remains the compatibility default for historical comparisons. New training runs fit scalers only on the chronological fit window, so their numerical outputs can differ from earlier runs.

## Explore the v9 return model

The [v9 notebook](examples/notebooks/stock_prediction_lstm_v9.ipynb) trains a shared LSTM with return, direction, and ordered quantile heads. It saves the model, scalers, input-data hash, test metrics, future forecast, and inference settings. The future forecast includes independent raw and anchored recursive paths. The model remains opt-in because the current held-out FTSE run did not beat the zero-return baseline on RMSE; see [model semantics](docs/MODELS.md).

## Benchmark saved predictions

Given an aligned CSV such as:

~~~text
Date,Actual,LSTM,MultiTask
2026-01-05,100.0,100.4,100.2
2026-01-06,101.0,100.8,101.1
...
~~~

run:

~~~bash
quant-forecast benchmark \
  --csv results.csv \
  --actual-col Actual \
  --prediction-col LSTM \
  --prediction-col MultiTask
~~~

The output includes RMSE, MAE, directional accuracy and RMSE skill versus the previous-price naive baseline.
<img width="1400" height="500" alt="image" src="https://github.com/user-attachments/assets/cc156634-048b-4c9c-a959-17e6ef22cc55" />

## Financial Forecasting Arena

For a multi-asset or multi-horizon comparison, use the arena. It consumes **already out-of-sample** predictions rather than training models inside the evaluator, keeping model fitting and evaluation cleanly separated.

~~~bash
quant-forecast arena \
  --csv benchmark.csv \
  --actual-col Actual \
  --prediction-col LSTM \
  --prediction-col MultiTask \
  --group-col Ticker \
  --group-col Horizon \
  --time-col Date \
  --previous-actual-col OriginClose \
  --cost-bps 5 \
  --output reports/generated/arena.html
~~~

The arena produces a self-contained HTML report and a table containing:

- RMSE and MAE;
- directional accuracy;
- RMSE skill versus the previous-price naive forecast;
- annualised directional-strategy return and volatility;
- Sharpe-like diagnostic;
- maximum drawdown;
- average turnover;
- explicit transaction-cost assumption.

Published benchmark results belong under [benchmarks/](benchmarks/README.md) and should never be generated from training observations or a hand-picked single run.

## Repository architecture

~~~text
.
├── quant_forecast_lab/                     # maintained, testable research utilities
│   ├── arena.py
│   ├── backtest.py
│   ├── benchmark.py
│   ├── config.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── reproducibility.py
│   ├── uncertainty.py
│   └── validation.py
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── METHODOLOGY.md
│   └── assets/legacy/
├── examples/
│   └── notebooks/
├── stock_prediction_*.py        # compatibility entry points
├── pyproject.toml
└── environment.yml
~~~

The migration is deliberately compatibility-first: stable research components are introduced around the historical scripts rather than breaking old commands through a big-bang move.

See [Architecture](docs/ARCHITECTURE.md) and [Methodology](docs/METHODOLOGY.md).

See also the [Model catalogue](docs/MODELS.md) for the exact semantics of v1-v9.

## Evaluation protocol

The training path separates three roles:

1. **Training** — optimise model weights.
2. **Validation** — early stopping and learning-rate decisions.
3. **Test** — final out-of-sample evaluation only.

The fixed date split remains available for compatibility. For research comparisons, prefer chronological expanding windows:

~~~python
from quant_forecast_lab.validation import expanding_window_splits

folds = expanding_window_splits(
    n_samples=2500,
    min_train_size=1250,
    validation_size=125,
    test_size=125,
    gap=1,
)
~~~

The splitter is intentionally independent of TensorFlow so the evaluation protocol can be tested without downloading market data.

## Uncertainty: scenario paths vs calibrated intervals

The inference implementation can generate stochastic trajectories by perturbing the model forecast using volatility estimated from recent history. These are **scenario paths**, not automatically calibrated probability statements.

Where enough held-out one-step prediction residuals are available, inference reports a symmetric **one-step residual band**. It has no validated coverage guarantee for recursive multi-session forecasts. The underlying conformal primitive is also exposed directly for experiments whose calibration and evaluation observations satisfy its assumptions:

~~~python
from quant_forecast_lab.uncertainty import symmetric_conformal_interval

lower, upper = symmetric_conformal_interval(
    point_forecast,
    calibration_actual,
    calibration_predicted,
    coverage=0.90,
)
~~~

Empirical interval coverage should still be measured on observations that were not used to set the interval radius.

## Trading diagnostics

A low RMSE does not imply a profitable strategy. The research package includes an intentionally small directional diagnostic:

~~~python
from quant_forecast_lab.backtest import backtest_directional_strategy

metrics = backtest_directional_strategy(
    actual_returns,
    predicted_returns,
    transaction_cost_bps=5,
)
~~~

It reports annualised return/volatility, Sharpe, maximum drawdown and turnover. It is not an execution simulator and does not model market impact, borrow, financing or capacity.

## Reproducibility

The training CLI accepts an explicit seed. The shared helper seeds Python, NumPy and TensorFlow and requests deterministic TensorFlow operations where supported.

~~~python
from quant_forecast_lab.reproducibility import set_global_seed

set_global_seed(42)
~~~

Each run writes its configuration alongside the model artifacts, saves the exact close-price snapshot as `market_data.csv`, records its SHA-256 digest, and captures Python, TensorFlow, NumPy, Pandas and Git revision metadata when available.

## GPU notes

TensorFlow 2.18.1 works on CPU. For NVIDIA GPU acceleration, use Linux or WSL2 on Windows. Native-Windows TensorFlow GPU support ended after TensorFlow 2.10, so the old native-Windows CUDA 11.2 instructions are intentionally removed.

Use the current [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for driver and CUDA requirements.

## Historical material

The original notebook is retained at:

~~~text
examples/notebooks/stock_prediction_lstm.ipynb
~~~

Historical screenshots are retained under:

~~~text
docs/assets/legacy/
~~~

Generated ticker/date experiment directories are intentionally removed from the project root. They remain recoverable from Git history, while new runs are ignored by Git.

## Forecasting a saved run

A known-good v7 FTSE reference run is retained under `examples/runs/reference-v7-ftse`, so the inference workflow remains executable after generated runs were removed from the repository root.

~~~bash
python stock_prediction_forecasting.py \
  --run-folder examples/runs/reference-v7-ftse \
  --ticker ^FTSE \
  --calendar XLON \
  --forecast-days 30
~~~

The calendar parameter is passed to `exchange_calendars`, so the reference forecast uses actual London Stock Exchange sessions rather than treating every weekday as tradable. For US equities use `XNYS`; for continuous markets an appropriate always-open calendar can be supplied.

The command reads the saved run without modifying it. Forecast CSV, metadata and plot outputs go to a fresh ignored directory under `runs/forecasts/`. Use `--output-folder runs/my-forecast` to choose a destination; it must be outside the model run folder.

## Data

The compatibility scripts use [yfinance](https://pypi.org/project/yfinance/) as a convenient public-data source. Data quality, corporate actions, symbol history and survivorship assumptions remain the responsibility of each experiment.

## Scope

This project is a forecasting research environment. Results from one asset, period or validation split should not be interpreted as evidence that a model generalises to another market regime or that a trading strategy is deployable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Model contributions should include a reproducible configuration, a naive baseline and out-of-sample evidence.

## License

Apache License 2.0.
