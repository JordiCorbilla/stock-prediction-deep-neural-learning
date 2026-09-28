# Contributing

Contributions are welcome, especially when they improve reproducibility, evaluation quality, data handling or documentation.

## Development setup

~~~bash
python -m venv .venv
# activate the environment for your platform
python -m pip install -e ".[dev]"
pytest
ruff check quant_forecast_lab tests
~~~

## Research changes

For model changes, include:

1. the exact data split or walk-forward protocol;
2. at least one naive baseline;
3. the seed/configuration required to reproduce the run;
4. out-of-sample metrics;
5. transaction-cost assumptions when presenting strategy metrics.

Do not commit generated run directories, model binaries, caches or IDE metadata.

## Compatibility

The historical root scripts remain supported entry points. Refactors should preserve their public command-line behaviour unless a breaking change is explicitly proposed and documented.
