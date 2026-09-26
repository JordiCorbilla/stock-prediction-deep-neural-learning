# Architecture

The cleanup deliberately uses a **strangler-style migration** rather than moving every historical module at once.

## Compatibility layer

The existing root scripts remain the supported legacy entry points:

- stock_prediction_deep_learning.py
- stock_prediction_deep_learning_inference.py
- stock_prediction_numpy.py
- stock_prediction_lstm.py

This prevents old commands, notebooks and external links from breaking.

## Maintained research layer

The stocklab package contains dependency-light components that can be tested without downloading market data or starting TensorFlow:

- config.py — shared model/target compatibility rules;
- evaluation.py — forecast error, direction and naive-relative skill;
- validation.py — chronological expanding-window folds;
- uncertainty.py — distribution-free conformal intervals;
- backtest.py — explicit transaction-cost-aware diagnostics;
- benchmark.py — model comparison;
- reproducibility.py — deterministic seed setup;
- cli.py — research utilities.

Future refactors can move training, data and model implementations behind these stable interfaces one component at a time.

## Design rules

1. No test data may select epochs or hyperparameters.
2. Every price forecast should be compared with a last-observation baseline.
3. Probabilistic claims require calibration evidence.
4. A backtest is a diagnostic, not proof of deployable profitability.
5. Generated runs live outside source control.
6. Legacy entry points are preserved until an explicit major-version migration.
