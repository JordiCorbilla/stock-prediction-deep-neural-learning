"""Small explicit backtesting helpers for forecast diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StrategyMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    average_turnover: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def backtest_directional_strategy(
    actual_returns,
    predicted_returns,
    *,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> StrategyMetrics:
    actual = np.asarray(actual_returns, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_returns, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Return arrays must have the same length.")
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) < 2:
        raise ValueError("At least two finite observations are required.")

    position = np.sign(predicted)
    previous = np.concatenate(([0.0], position[:-1]))
    turnover = np.abs(position - previous)
    costs = turnover * (float(transaction_cost_bps) / 10_000.0)
    strategy_returns = position * actual - costs

    mean_return = float(np.mean(strategy_returns))
    volatility = float(np.std(strategy_returns, ddof=1))
    annualized_return = mean_return * periods_per_year
    annualized_volatility = volatility * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_volatility if annualized_volatility else 0.0

    equity = np.cumprod(1.0 + strategy_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    return StrategyMetrics(
        annualized_return=float(annualized_return),
        annualized_volatility=float(annualized_volatility),
        sharpe=float(sharpe),
        max_drawdown=float(np.min(drawdown)),
        average_turnover=float(np.mean(turnover)),
    )
