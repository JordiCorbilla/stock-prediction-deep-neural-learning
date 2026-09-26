# Published benchmarks

This directory is reserved for reproducible, out-of-sample benchmark outputs.

Do not commit a leaderboard produced from training data or a single hand-picked run.

## Arena input

The arena consumes an aligned CSV containing actual prices and one or more model predictions. Optional grouping columns can represent ticker and forecast horizon.

Example:

~~~text
Date,Ticker,Horizon,Actual,Naive,LSTM,MultiTask
2026-01-05,SPY,1,100.0,99.7,100.2,100.1
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
  --cost-bps 5 \
  --output reports/generated/arena.html
~~~

The generated table includes forecast error, directional accuracy, naïve-relative RMSE skill, and simple transaction-cost-aware directional strategy diagnostics.
