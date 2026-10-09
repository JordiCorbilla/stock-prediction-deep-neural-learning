import numpy as np
import pytest

from quant_forecast_lab.icaif import enhanced_features, portfolio_target, sequence_features, trade_required


def test_features_cannot_see_future_steps():
    rng = np.random.default_rng(7)
    base = rng.normal(0, .1, (30, 20, 3)).astype("float32")
    full = enhanced_features(base)
    np.testing.assert_allclose(full[:, :10], enhanced_features(base[:, :10]), atol=1e-7)


def test_lookback_recomputes_features_and_vix_alignment():
    base = np.zeros((30, 20, 3), dtype="float32")
    result = sequence_features(base, 10, np.arange(11) + 20.)
    assert result.shape == (30, 10, 8)
    np.testing.assert_allclose(result[0, :, 6], np.arange(21, 31) / 40.)
    with pytest.raises(ValueError):
        sequence_features(base, 10, np.ones(10))


def test_allocations_and_no_trade_band():
    target = portfolio_target(np.arange(30.))
    assert np.isclose(target.sum(), .99) and target.max() < .30
    assert trade_required(np.zeros(30), target)
    assert not trade_required(target, target)
    with pytest.raises(ValueError):
        trade_required(target, target, float("nan"))
