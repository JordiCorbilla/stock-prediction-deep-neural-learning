# Forecasting methodology

## Baseline first

Price-level forecasts must be compared against the last-observation baseline:

~~~text
prediction(t) = actual(t-1)
~~~

The package reports:

~~~text
RMSE skill = 1 - RMSE(model) / RMSE(naive)
~~~

Positive skill indicates lower RMSE than the naive predictor; zero is equivalent; negative values are worse.

## Chronological validation

Random train/test shuffling is inappropriate for this project. Use chronological splits and, when comparing research variants, expanding-window walk-forward evaluation.

stocklab.validation.expanding_window_splits provides deterministic fold boundaries while leaving model training policy to the caller.

## Direction

Directional accuracy is measured relative to the information available immediately before the forecast:

~~~text
sign(predicted_price[t] - actual_price[t-1])
vs
sign(actual_price[t] - actual_price[t-1])
~~~

## Uncertainty

The historical stochastic trajectories are scenario simulations obtained by perturbing a point forecast with noise estimated from recent history. They should not be interpreted as calibrated prediction intervals.

stocklab.uncertainty provides a symmetric conformal interval based on held-out residuals. Coverage should still be measured empirically on out-of-sample data.

## Trading diagnostics

Prediction error and trading value are different questions. When a forecast is translated into a directional strategy, report transaction costs, turnover, Sharpe-like diagnostics and drawdown in addition to forecasting metrics.
