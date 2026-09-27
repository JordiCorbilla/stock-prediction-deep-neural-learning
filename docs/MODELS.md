# Model catalogue

The project keeps historical model identifiers for reproducibility while documenting what each one actually represents.

| Version | Target / architecture | Status |
| --- | --- | --- |
| v1 | stacked LSTM price-level model | legacy |
| v2 | compact LSTM price-level model | legacy |
| v3 | compact LSTM delta target | legacy |
| v4 | larger LSTM with learning-rate reduction | legacy |
| v5 | multi-horizon delta output | experimental |
| v6 | trend-residual output | experimental |
| v7 | two independent LSTMs: direction classifier + magnitude regressor | compatibility default |
| v8 | shared LSTM encoder with direction and magnitude output heads | opt-in research model |
| v9 | shared LSTM encoder with return, direction and ordered return-quantile heads | opt-in research model |

## v7

v7 is intentionally retained as the default so existing users are not silently moved to a new architecture. It trains two independent networks and reconstructs a signed price delta from:

- probability of an upward move;
- predicted absolute move magnitude.

The two networks do **not** share weights.

## v8

v8 is the first true multi-task version:

~~~text
                         +--> direction: sigmoid
input --> LSTM --> LSTM -|
                         +--> magnitude: softplus
~~~

Both targets share the same temporal representation. This reduces duplicated representation learning and makes "two heads" an accurate description.

v8 is not declared superior merely because it is newer. It should be compared with v7 and naive baselines using chronological out-of-sample evaluation.

## v9

v9 predicts next-session log return with a shared LSTM and three heads: expected return, up-move probability, and ordered q10/q50/q90 return estimates. The return head is used to reconstruct the next close from the **observed** prior close during test evaluation. Future forecasts recursively feed model returns into the input window. The optional long-horizon anchor is a separately recorded heuristic; `Predicted_Price_Unanchored` uses an independent raw recursive window.

Saved v9 models can be loaded for inference with `tf.keras.models.load_model(path, compile=False)`. To resume training with the serialized quantile loss, use `stock_prediction_deep_learning.load_v9_model(path, compile_model=True)`.

Quantile names express nominal levels, not guaranteed calibration. The saved `return_forecast_metrics.json` records empirical CDF values and q10–q90 coverage on the untouched test period. A September 2026 FTSE run had **negative** RMSE skill against a zero-return baseline; it is an architecture demonstration, not a claim of superior forecast performance.

## Model-selection rule

A model should not become the compatibility default because it wins one ticker or one split. Promotion requires:

1. positive naive-relative forecast skill across multiple assets and periods;
2. stable directional performance;
3. no leakage from final test observations;
4. reproducible seeds/configuration;
5. evidence that gains survive realistic transaction-cost assumptions when a trading interpretation is presented.
