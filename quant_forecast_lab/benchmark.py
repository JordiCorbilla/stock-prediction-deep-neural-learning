"""Model-vs-naive benchmarking for aligned price forecasts."""

from __future__ import annotations

import pandas as pd

from .evaluation import evaluate_price_forecast


def benchmark_frame(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    prediction_cols: list[str],
    previous_actual_col: str | None = None,
) -> pd.DataFrame:
    if actual_col not in frame:
        raise KeyError(f"Missing actual column: {actual_col}")
    for column in prediction_cols:
        if column not in frame:
            raise KeyError(f"Missing prediction column: {column}")

    work = frame.copy()
    if previous_actual_col is None:
        previous = work[actual_col].shift(1)
    else:
        if previous_actual_col not in work:
            raise KeyError(f"Missing previous-actual column: {previous_actual_col}")
        previous = work[previous_actual_col]

    rows = []
    for column in prediction_cols:
        aligned = pd.DataFrame(
            {"actual": work[actual_col], "predicted": work[column], "previous": previous}
        ).dropna()
        metrics = evaluate_price_forecast(
            aligned["actual"],
            aligned["predicted"],
            aligned["previous"],
        )
        rows.append({"model": column, **metrics.as_dict()})

    return pd.DataFrame(rows).set_index("model").sort_values(
        "rmse_skill_vs_naive", ascending=False
    )
