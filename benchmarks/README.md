# Published benchmarks

This directory is reserved for reproducible, out-of-sample benchmark outputs.

Do not commit a leaderboard produced from training data or a single hand-picked run.

The walk-forward harness can use a frozen `Date,Close` CSV with `--input-csv`. Each new run saves its input snapshot and SHA-256 in the summary. To reproduce a run, retain that CSV with the metrics and predictions. Live `yfinance` history may be revised later.

## Arena input

The arena consumes an aligned CSV containing actual prices and one or more model predictions. Optional grouping columns can represent ticker and forecast horizon.

Example:

~~~text
Date,Ticker,Horizon,OriginClose,Actual,LSTM,MultiTask
2026-01-05,SPY,3,99.7,100.0,100.2,100.1
...
~~~

## Run

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

The generated table includes forecast error, directional accuracy, naïve-relative RMSE skill, and simple transaction-cost-aware directional strategy diagnostics.

`OriginClose` must be the price available when each forecast was made. For horizons above one session, shifting the target-date actual close would read a future price and distort every baseline-relative and directional statistic, so the arena rejects that input without an explicit origin close.

Annualised strategy diagnostics default to 252 periods per year; pass `--periods-per-year 365` for daily continuous-market data.
