"""Leakage-aware forecast evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastMetrics:
    rmse: float
    mae: float
    directional_accuracy: float
    rmse_skill_vs_naive: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _arrays(*values):
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("All arrays must have the same length.")
    mask = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    return [value[mask] for value in arrays]


def rmse(actual, predicted) -> float:
    actual, predicted = _arrays(actual, predicted)
    if len(actual) == 0:
        raise ValueError("No finite observations available.")
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mae(actual, predicted) -> float:
    actual, predicted = _arrays(actual, predicted)
    if len(actual) == 0:
        raise ValueError("No finite observations available.")
    return float(np.mean(np.abs(actual - predicted)))


def directional_accuracy(actual, predicted, previous_actual) -> float:
    actual, predicted, previous_actual = _arrays(actual, predicted, previous_actual)
    if len(actual) == 0:
        raise ValueError("No finite observations available.")
    actual_direction = np.sign(actual - previous_actual)
    predicted_direction = np.sign(predicted - previous_actual)
    return float(np.mean(actual_direction == predicted_direction))


def evaluate_price_forecast(actual, predicted, previous_actual) -> ForecastMetrics:
    """Evaluate prices against the explicit last-observation naive baseline."""
    actual, predicted, previous_actual = _arrays(actual, predicted, previous_actual)
    if len(actual) == 0:
        raise ValueError("No finite observations available.")

    model_rmse = rmse(actual, predicted)
    naive_rmse = rmse(actual, previous_actual)
    if naive_rmse == 0:
        skill = 0.0 if model_rmse == 0 else float("-inf")
    else:
        skill = 1.0 - (model_rmse / naive_rmse)

    return ForecastMetrics(
        rmse=model_rmse,
        mae=mae(actual, predicted),
        directional_accuracy=directional_accuracy(actual, predicted, previous_actual),
        rmse_skill_vs_naive=float(skill),
    )
