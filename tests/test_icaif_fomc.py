import numpy as np
import pytest

from quant_forecast_lab.icaif_fomc import append_calendar_context, calendar_context


def test_weekend_and_pre_announcement_flag():
    sessions = ["2025-01-24", "2025-01-27", "2025-01-28", "2025-01-29"]
    events = ["2025-01-29"]
    np.testing.assert_allclose(calendar_context("2025-01-28", sessions, events), [0.05, 1])
    np.testing.assert_array_equal(calendar_context("2025-01-29", sessions, events), [0, 1])
    np.testing.assert_allclose(calendar_context("2025-01-24", sessions, events), [0.15, 0])


def test_added_context_preserves_stock_features_and_has_no_price_dependency():
    original = np.arange(30 * 20 * 6, dtype="float32").reshape(30, 20, 6)
    result = append_calendar_context(original, [0.05, 1])
    np.testing.assert_array_equal(result[:, :, :6], original)
    assert result.shape == (30, 20, 8)
    np.testing.assert_allclose(result[:, :, 6], 0.05)
    np.testing.assert_array_equal(result[:, :, 7], 1)


def test_missing_future_calendar_is_an_error():
    with pytest.raises(ValueError, match="unavailable"):
        calendar_context("2025-01-29", ["2025-01-29"], ["2025-01-28"])
