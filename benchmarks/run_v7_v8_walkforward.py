"""Rigorous one-step walk-forward benchmark for v7, v8 and a naive baseline.

The benchmark intentionally does not reuse fitted legacy artifacts. Each fold:
1. downloads one adjusted close series and freezes it to CSV;
2. fits scalers using training observations only;
3. trains v7 and v8 with chronological validation and shuffle=False;
4. scores untouched one-step-ahead test observations;
5. compares both models with the previous-close naive forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from quant_forecast_lab.backtest import backtest_directional_strategy
from quant_forecast_lab.evaluation import evaluate_price_forecast
from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.reproducibility import set_global_seed
from stock_prediction_lstm import LongShortTermMemory

TEST_YEARS = (2020, 2022, 2024, 2026)


def _download_close(ticker: str, start: str, end: str, attempts: int = 3) -> pd.Series:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if frame.empty:
                raise RuntimeError(f"No data returned for {ticker}")
            close = frame["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close.name = "Close"
            if len(close) < 1000:
                raise RuntimeError(f"Insufficient history for {ticker}: {len(close)} observations")
            return close
        except Exception as exc:  # network/API retry boundary
            last_error = exc
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Unable to download {ticker}: {last_error}") from last_error


def _load_close_snapshot(path: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    if list(frame.columns) != ["Close"]:
        raise ValueError("Input snapshot must contain exactly one Close column.")
    close = pd.to_numeric(frame["Close"], errors="raise").astype(float)
    if close.isna().any() or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Input Close prices must be finite and positive.")
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Input dates must be unique and chronologically sorted.")
    return close


def _samples(
    close: pd.Series,
    target_dates: pd.DatetimeIndex,
    *,
    window: int,
    input_scaler: MinMaxScaler,
    magnitude_scaler: MinMaxScaler,
):
    values = close.to_numpy(dtype=float)
    dates = close.index
    position = {date: idx for idx, date in enumerate(dates)}

    x: list[np.ndarray] = []
    y_direction: list[float] = []
    y_magnitude: list[float] = []
    meta: list[tuple[pd.Timestamp, float, float, float]] = []

    for date in target_dates:
        idx = position.get(date)
        if idx is None or idx < window:
            continue
        previous = values[idx - 1]
        actual = values[idx]
        delta = actual - previous
        history = values[idx - window : idx].reshape(-1, 1)
        scaled_history = input_scaler.transform(history)
        scaled_magnitude = magnitude_scaler.transform([[abs(delta)]])[0, 0]

        x.append(scaled_history)
        y_direction.append(float(delta > 0.0))
        y_magnitude.append(float(scaled_magnitude))
        meta.append((date, actual, previous, delta))

    if not x:
        return (
            np.empty((0, window, 1), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y_direction, dtype=np.float32),
        np.asarray(y_magnitude, dtype=np.float32),
        meta,
    )


def _callbacks():
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        ),
    ]


def _strategy_metrics(actual_delta, predicted_delta, previous, cost_bps: float, periods_per_year=252):
    actual_returns = np.asarray(actual_delta, dtype=float) / np.asarray(previous, dtype=float)
    predicted_returns = np.asarray(predicted_delta, dtype=float) / np.asarray(previous, dtype=float)
    return backtest_directional_strategy(
        actual_returns,
        predicted_returns,
        transaction_cost_bps=cost_bps,
        periods_per_year=periods_per_year,
    )


def _ic(actual_delta, predicted_delta) -> float:
    actual = np.asarray(actual_delta, dtype=float)
    predicted = np.asarray(predicted_delta, dtype=float)
    if len(actual) < 3 or np.std(actual) == 0.0 or np.std(predicted) == 0.0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def _evaluate_predictions(actual, predicted_price, previous, actual_delta, predicted_delta, cost_bps, periods_per_year=252):
    forecast = evaluate_price_forecast(actual, predicted_price, previous)
    strategy = _strategy_metrics(actual_delta, predicted_delta, previous, cost_bps, periods_per_year)
    return {
        **forecast.as_dict(),
        "delta_ic": _ic(actual_delta, predicted_delta),
        **strategy.as_dict(),
    }


def _fit_predict_v7(helper, x_train, y_dir_train, y_mag_train, x_val, y_dir_val, y_mag_val, x_test, magnitude_scaler, epochs, batch_size):
    direction_model = helper._create_model_v7(x_train, output_units=1, activation="sigmoid")
    direction_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    direction_model.fit(
        x_train,
        y_dir_train,
        validation_data=(x_val, y_dir_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=_callbacks(),
        verbose=0,
    )

    magnitude_model = helper._create_model_v7(x_train, output_units=1, activation=None)
    magnitude_model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())
    magnitude_model.fit(
        x_train,
        y_mag_train,
        validation_data=(x_val, y_mag_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=_callbacks(),
        verbose=0,
    )

    direction_probability = direction_model.predict(x_test, verbose=0).reshape(-1)
    magnitude_scaled = magnitude_model.predict(x_test, verbose=0).reshape(-1, 1)
    magnitude = magnitude_scaler.inverse_transform(magnitude_scaled).reshape(-1)
    sign = np.where(direction_probability >= 0.5, 1.0, -1.0)
    return sign * np.maximum(magnitude, 0.0)


def _fit_predict_v8(helper, x_train, y_dir_train, y_mag_train, x_val, y_dir_val, y_mag_val, x_test, magnitude_scaler, epochs, batch_size):
    model = helper.create_multitask_model(x_train)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={"direction": "binary_crossentropy", "magnitude": Huber()},
        metrics={"direction": ["accuracy"], "magnitude": [tf.keras.metrics.MeanSquaredError(name="MSE")]},
    )
    model.fit(
        x_train,
        {"direction": y_dir_train, "magnitude": y_mag_train},
        validation_data=(x_val, {"direction": y_dir_val, "magnitude": y_mag_val}),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=_callbacks(),
        verbose=0,
    )
    output = model.predict(x_test, verbose=0)
    direction_probability = np.asarray(output["direction"]).reshape(-1)
    magnitude_scaled = np.asarray(output["magnitude"]).reshape(-1, 1)
    magnitude = magnitude_scaler.inverse_transform(magnitude_scaled).reshape(-1)
    sign = np.where(direction_probability >= 0.5, 1.0, -1.0)
    return sign * np.maximum(magnitude, 0.0)


def run_benchmark(args) -> dict:
    set_global_seed(args.seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(args.tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.tf_threads)
    except RuntimeError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.label.replace("^", "").replace("-", "").replace(" ", "_")

    close = _load_close_snapshot(args.input_csv) if args.input_csv else _download_close(args.ticker, args.start, args.end)
    snapshot = close.to_frame()
    snapshot_path = output_dir / f"market_data_{prefix}.csv"
    snapshot.to_csv(snapshot_path)
    input_sha256 = sha256_file(args.input_csv) if args.input_csv else sha256_file(snapshot_path)

    rows: list[dict] = []
    prediction_rows: list[dict] = []
    helper = LongShortTermMemory(str(output_dir))

    for test_year in TEST_YEARS:
        validation_start = pd.Timestamp(test_year - 1, 1, 1)
        test_start = pd.Timestamp(test_year, 1, 1)
        test_end = pd.Timestamp(test_year + 1, 1, 1)
        if test_year == 2026:
            test_end = min(test_end, pd.Timestamp(args.end))

        train_dates = close.index[close.index < validation_start]
        validation_dates = close.index[(close.index >= validation_start) & (close.index < test_start)]
        test_dates = close.index[(close.index >= test_start) & (close.index < test_end)]

        if len(train_dates) < args.min_train_observations or len(validation_dates) < 100 or len(test_dates) < 100:
            print(
                f"SKIP {args.ticker} {test_year}: "
                f"train={len(train_dates)} val={len(validation_dates)} test={len(test_dates)}"
            )
            continue

        input_scaler = MinMaxScaler()
        input_scaler.fit(close.loc[train_dates].to_numpy().reshape(-1, 1))

        train_delta = close.diff().loc[train_dates].dropna()
        magnitude_scaler = MinMaxScaler()
        magnitude_scaler.fit(train_delta.abs().to_numpy().reshape(-1, 1))

        x_train, y_dir_train, y_mag_train, _ = _samples(
            close,
            train_dates,
            window=args.window,
            input_scaler=input_scaler,
            magnitude_scaler=magnitude_scaler,
        )
        x_val, y_dir_val, y_mag_val, _ = _samples(
            close,
            validation_dates,
            window=args.window,
            input_scaler=input_scaler,
            magnitude_scaler=magnitude_scaler,
        )
        x_test, _, _, test_meta = _samples(
            close,
            test_dates,
            window=args.window,
            input_scaler=input_scaler,
            magnitude_scaler=magnitude_scaler,
        )

        if min(len(x_train), len(x_val), len(x_test)) == 0:
            raise RuntimeError(f"Empty supervised split for {args.ticker} {test_year}")

        actual = np.asarray([item[1] for item in test_meta], dtype=float)
        previous = np.asarray([item[2] for item in test_meta], dtype=float)
        actual_delta = np.asarray([item[3] for item in test_meta], dtype=float)
        dates = [item[0] for item in test_meta]

        naive_price = previous.copy()
        naive_delta = np.zeros_like(previous)
        naive_metrics = _evaluate_predictions(
            actual,
            naive_price,
            previous,
            actual_delta,
            naive_delta,
            args.cost_bps,
            365 if args.ticker.upper() == "BTC-USD" else 252,
        )
        rows.append(
            {
                "ticker": args.ticker,
                "label": args.label,
                "test_year": test_year,
                "model": "naive",
                "observations": len(actual),
                "train_observations": len(x_train),
                "validation_observations": len(x_val),
                **naive_metrics,
            }
        )

        for date, act, prev in zip(dates, actual, previous, strict=True):
            prediction_rows.append(
                {
                    "Date": date,
                    "Ticker": args.ticker,
                    "TestYear": test_year,
                    "Model": "naive",
                    "Actual": act,
                    "Previous": prev,
                    "Predicted": prev,
                }
            )

        for model_name in ("v7", "v8"):
            tf.keras.backend.clear_session()
            set_global_seed(args.seed + test_year + (0 if model_name == "v7" else 10000))
            started = time.perf_counter()

            if model_name == "v7":
                predicted_delta = _fit_predict_v7(
                    helper,
                    x_train,
                    y_dir_train,
                    y_mag_train,
                    x_val,
                    y_dir_val,
                    y_mag_val,
                    x_test,
                    magnitude_scaler,
                    args.epochs,
                    args.batch_size,
                )
            else:
                predicted_delta = _fit_predict_v8(
                    helper,
                    x_train,
                    y_dir_train,
                    y_mag_train,
                    x_val,
                    y_dir_val,
                    y_mag_val,
                    x_test,
                    magnitude_scaler,
                    args.epochs,
                    args.batch_size,
                )

            predicted_price = previous + predicted_delta
            metrics = _evaluate_predictions(
                actual,
                predicted_price,
                previous,
                actual_delta,
                predicted_delta,
                args.cost_bps,
                365 if args.ticker.upper() == "BTC-USD" else 252,
            )
            elapsed = time.perf_counter() - started

            row = {
                "ticker": args.ticker,
                "label": args.label,
                "test_year": test_year,
                "model": model_name,
                "observations": len(actual),
                "train_observations": len(x_train),
                "validation_observations": len(x_val),
                "seconds": elapsed,
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

            for date, act, prev, pred in zip(dates, actual, previous, predicted_price, strict=True):
                prediction_rows.append(
                    {
                        "Date": date,
                        "Ticker": args.ticker,
                        "TestYear": test_year,
                        "Model": model_name,
                        "Actual": act,
                        "Previous": prev,
                        "Predicted": pred,
                    }
                )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"No valid benchmark folds for {args.ticker}")

    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output_dir / f"metrics_{prefix}.csv", index=False)
    predictions.to_csv(output_dir / f"predictions_{prefix}.csv", index=False)

    summaries = []
    for model_name, group in predictions.groupby("Model", sort=False):
        actual = group["Actual"].to_numpy(float)
        predicted = group["Predicted"].to_numpy(float)
        previous = group["Previous"].to_numpy(float)
        actual_delta = actual - previous
        predicted_delta = predicted - previous
        combined = _evaluate_predictions(
            actual,
            predicted,
            previous,
            actual_delta,
            predicted_delta,
            args.cost_bps,
            365 if args.ticker.upper() == "BTC-USD" else 252,
        )
        fold_rows = metrics[metrics["model"] == model_name]
        combined.update(
            {
                "ticker": args.ticker,
                "label": args.label,
                "model": model_name,
                "observations": int(len(group)),
                "folds": int(len(fold_rows)),
                "median_fold_skill": float(fold_rows["rmse_skill_vs_naive"].median()),
                "positive_skill_folds": int((fold_rows["rmse_skill_vs_naive"] > 0).sum()),
            }
        )
        summaries.append(combined)

    summary = {
        "protocol": {
            "ticker": args.ticker,
            "label": args.label,
            "start": args.start,
            "end": args.end,
            "test_years": list(TEST_YEARS),
            "window": args.window,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "transaction_cost_bps": args.cost_bps,
            "auto_adjust": True,
            "input_source": args.input_csv or "yfinance download",
            "input_sha256": input_sha256,
            "market_data_file": snapshot_path.name,
            "periods_per_year": 365 if args.ticker.upper() == "BTC-USD" else 252,
            "forecast_type": "rolling one-step-ahead using observed history",
            "shuffle": False,
        },
        "models": summaries,
    }
    summary_path = output_dir / f"summary_{prefix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    print("BENCHMARK_SUMMARY_START")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print("BENCHMARK_SUMMARY_END")
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-09-27")
    parser.add_argument("--input-csv", help="Frozen Date,Close snapshot; skips the mutable Yahoo download.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-train-observations", type=int, default=700)
    parser.add_argument("--tf-threads", type=int, default=2)
    return parser


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
