import numpy as np

from quant_forecast_lab.evaluation import evaluate_price_forecast


def test_perfect_forecast_beats_naive():
    previous = np.array([100.0, 101.0, 100.0, 103.0])
    actual = np.array([101.0, 100.0, 103.0, 104.0])
    metrics = evaluate_price_forecast(actual, actual.copy(), previous)

    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.directional_accuracy == 1.0
    assert metrics.rmse_skill_vs_naive == 1.0


def test_naive_equivalent_has_zero_skill():
    previous = np.array([100.0, 101.0, 100.0, 103.0])
    actual = np.array([101.0, 100.0, 103.0, 104.0])
    metrics = evaluate_price_forecast(actual, previous, previous)

    assert metrics.rmse_skill_vs_naive == 0.0


def test_zero_error_naive_has_undefined_relative_skill_for_wrong_model():
    actual = np.array([100.0, 100.0])
    metrics = evaluate_price_forecast(actual, np.array([101.0, 101.0]), actual)
    assert metrics.rmse_skill_vs_naive is None
