"""Runtime checks that use the full research dependency stack."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from quant_forecast_lab.legacy_keras import load_legacy_h5
from stock_prediction_class import StockPrediction
from stock_prediction_deep_learning import _chronological_validation_split
from stock_prediction_deep_learning_inference import _future_dates
from stock_prediction_numpy import StockData


def _stock(tmp_path, validation_date):
    return StockPrediction(
        "TEST", pd.Timestamp("2024-01-01"), validation_date,
        str(tmp_path), "", 1, 3, "test", 2,
    )


def test_return_scaler_excludes_validation_extreme(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=40)
    closes = np.full(40, 100.0)
    closes[1:30] = 100.0 * np.exp(np.arange(1, 30) * 0.001)
    closes[27] = 1000.0
    closes[28] = 100.0
    frame = pd.DataFrame({"Close": closes}, index=dates)
    frame.index.name = "Date"
    stock = _stock(tmp_path, dates[30])
    data = StockData(stock)
    monkeypatch.setattr(data, "_download_close_frame", lambda _: frame)

    (x_train, _), _, _ = data.download_transform_to_numpy(
        3, str(tmp_path), use_returns=True, validation_fraction=0.2,
    )
    fit_count = StockData._fit_sample_count(29 - 3, 0.2)
    fit_returns = np.log(frame["Close"]).diff().dropna().iloc[:3 + fit_count]
    assert np.isclose(data.get_min_max().data_max_[0], fit_returns.max())
    assert data.get_min_max().data_max_[0] < 0.01
    assert len(x_train) == 26


@pytest.mark.parametrize("version", ["delta_horizon_two", "v7_direction"])
def test_input_scaler_stops_at_last_fit_input(tmp_path, monkeypatch, version):
    dates = pd.bdate_range("2024-01-01", periods=40)
    closes = 100.0 + np.arange(40, dtype=float)
    boundary = 22 if version == "delta_horizon_two" else 23
    closes[boundary] = 1000.0  # First close outside all fit input windows.
    frame = pd.DataFrame({"Close": closes}, index=dates)
    frame.index.name = "Date"
    stock = _stock(tmp_path, dates[30])
    data = StockData(stock)
    monkeypatch.setattr(data, "_download_close_frame", lambda _: frame)

    if version == "delta_horizon_two":
        data.download_transform_to_numpy(
            3, str(tmp_path), use_deltas=True, forecast_horizon=2,
            validation_fraction=0.2,
        )
    else:
        data.prepare_delta_direction_data(3, dates[30], validation_fraction=0.2)
    assert data.get_input_scaler().data_max_[0] == 100.0 + boundary - 1


def test_legacy_h5_weights_load_under_keras_3(tmp_path):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(3, 1)),
        tf.keras.layers.LSTM(4, return_sequences=True),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(2),
        tf.keras.layers.Dense(1),
    ])
    path = tmp_path / "model_weights.h5"
    model.save(path)
    restored = load_legacy_h5(path)
    sample = np.arange(6, dtype="float32").reshape(2, 3, 1)
    np.testing.assert_allclose(restored.predict(sample, verbose=0), model.predict(sample, verbose=0), rtol=1e-6)


def test_reference_v7_models_are_present_and_loadable():
    folder = Path("examples/runs/reference-v7-ftse")
    with (folder / "model_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    for name in ("model_direction.keras", "model_magnitude.keras"):
        path = folder / name
        assert path.is_file(), name
        model = tf.keras.models.load_model(path, compile=False)
        assert model.input_shape[1] == config["time_steps"]


def test_forecast_horizon_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        _future_dates(pd.Timestamp("2026-09-27"), 0, True, "XLON")


def test_multi_horizon_validation_purges_overlapping_targets():
    (fit, validation), = _chronological_validation_split(np.arange(20), fraction=0.2, purge=2)
    assert fit[-1] == 13
    assert validation[0] == 16
