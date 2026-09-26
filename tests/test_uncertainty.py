import numpy as np

from quant_forecast_lab.uncertainty import conformal_radius, symmetric_conformal_interval


def test_conformal_interval_is_symmetric_and_contains_point():
    actual = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    predicted = np.array([10.1, 10.8, 12.2, 12.8, 14.1])
    point = np.array([15.0, 16.0])

    radius = conformal_radius(actual, predicted, coverage=0.8)
    lower, upper = symmetric_conformal_interval(point, actual, predicted, coverage=0.8)

    assert radius >= 0
    assert np.all(lower <= point)
    assert np.all(upper >= point)
    np.testing.assert_allclose(point - lower, upper - point)
