"""Cross-asset forecasting arena for already out-of-sample predictions."""

from __future__ import annotations

import html as html_lib
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest_directional_strategy
from .evaluation import evaluate_price_forecast


FORECAST_COLUMNS = [
    "rmse",
    "mae",
    "directional_accuracy",
    "rmse_skill_vs_naive",
]

STRATEGY_COLUMNS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "average_turnover",
]


def arena_frame(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    prediction_cols: list[str],
    group_cols: list[str] | None = None,
    time_col: str | None = None,
    previous_actual_col: str | None = None,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Evaluate multiple models across optional ticker/horizon groups.

    The input is assumed to contain out-of-sample predictions. The function
    deliberately does not train or select models so evaluation remains
    separated from model fitting.
    """
    group_cols = list(group_cols or [])
    required = [actual_col, *prediction_cols, *group_cols]
    if time_col:
        required.append(time_col)
    if previous_actual_col:
        required.append(previous_actual_col)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("Missing columns: " + ", ".join(missing))

    grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, sort=True, dropna=False)
    rows: list[dict[str, object]] = []

    for key, group in grouped:
        if time_col:
            group = group.sort_values(time_col)
        keys = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_cols, keys, strict=True))

        if previous_actual_col:
            previous = group[previous_actual_col]
        else:
            previous = group[actual_col].shift(1)

        for model in prediction_cols:
            aligned = pd.DataFrame(
                {
                    "actual": pd.to_numeric(group[actual_col], errors="coerce"),
                    "predicted": pd.to_numeric(group[model], errors="coerce"),
                    "previous": pd.to_numeric(previous, errors="coerce"),
                }
            ).dropna()
            if len(aligned) < 2:
                continue

            forecast = evaluate_price_forecast(
                aligned["actual"],
                aligned["predicted"],
                aligned["previous"],
            )

            denom = aligned["previous"].replace(0.0, np.nan)
            actual_returns = aligned["actual"].div(denom).sub(1.0)
            predicted_returns = aligned["predicted"].div(denom).sub(1.0)
            valid_returns = pd.DataFrame(
                {"actual": actual_returns, "predicted": predicted_returns}
            ).dropna()

            strategy_values = {column: np.nan for column in STRATEGY_COLUMNS}
            if len(valid_returns) >= 2:
                strategy = backtest_directional_strategy(
                    valid_returns["actual"],
                    valid_returns["predicted"],
                    transaction_cost_bps=transaction_cost_bps,
                )
                strategy_values = strategy.as_dict()

            rows.append(
                {
                    **group_values,
                    "model": model,
                    "observations": len(aligned),
                    **forecast.as_dict(),
                    **strategy_values,
                }
            )

    if not rows:
        columns = [*group_cols, "model", "observations", *FORECAST_COLUMNS, *STRATEGY_COLUMNS]
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    sort_cols = [*group_cols, "rmse_skill_vs_naive"]
    ascending = [True] * len(group_cols) + [False]
    return result.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def render_html_report(metrics: pd.DataFrame, output_path, *, title: str = "Financial Forecasting Arena") -> Path:
    """Write a self-contained HTML report from arena metrics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    numeric = metrics.copy()
    for column in FORECAST_COLUMNS + STRATEGY_COLUMNS:
        if column in numeric:
            numeric[column] = numeric[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )

    table = numeric.to_html(index=False, escape=True, border=0, classes="arena")
    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 40px auto; padding: 0 24px; }}
h1 {{ margin-bottom: 0.25rem; }}
p {{ max-width: 900px; line-height: 1.55; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ position: sticky; top: 0; background: white; }}
code {{ background: #f5f5f5; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p>
Metrics are computed from supplied out-of-sample predictions. RMSE skill is relative to the
last-observation naive forecast. Strategy diagnostics use forecast direction and include
the configured transaction cost; they are diagnostics rather than an execution simulation.
</p>
{table}
</body>
</html>
"""
    path.write_text(report_html, encoding="utf-8")
    return path
