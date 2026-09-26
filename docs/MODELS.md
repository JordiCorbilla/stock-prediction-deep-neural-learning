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

## Model-selection rule

A model should not become the compatibility default because it wins one ticker or one split. Promotion requires:

1. positive naive-relative forecast skill across multiple assets and periods;
2. stable directional performance;
3. no leakage from final test observations;
4. reproducible seeds/configuration;
5. evidence that gains survive realistic transaction-cost assumptions when a trading interpretation is presented.
