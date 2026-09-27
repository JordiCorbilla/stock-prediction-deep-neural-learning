from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(value: str):
    return new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return new_code_cell(textwrap.dedent(value).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = [
        md(
            """
            # v9 — Return-first multi-task forecasting

            ## tl;dr

            This notebook is a **leakage-safe, runnable research example** for the proposed v9 direction: predict **next-session return, direction, and an uncertainty interval** rather than the absolute price level.

            The executed metrics below compare three out-of-sample forecasts on **SPY 2026 YTD**:

            1. **Zero-return baseline** — tomorrow's return = 0.
            2. **Ridge baseline** — a simple linear model on the same 30-session feature window.
            3. **v9 multi-task LSTM** — shared encoder with direction, expected-return, and quantile heads.

            The notebook deliberately does not claim an edge unless v9 beats the zero-return and Ridge baselines on held-out data.
            """
        ),
        md(
            r"""
            ## Context & Methods

            ### Architecture

            ```text
                             LAST 30 TRADING SESSIONS
                                      │
                     stationary / scale-aware features
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
       log return                  volatility                momentum
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                               train-only scaling
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │    SHARED ENCODER      │
                         │   LSTM 64 → LSTM 32    │
                         └───────────┬────────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
           DIRECTION HEAD      RETURN HEAD       QUANTILE HEAD
              P(up)            E[r(t+1)]        q10 / q50 / q90
                  │                  │                  │
                  └──────────────────┼──────────────────┘
                                     ▼
                             HELD-OUT 2026 TEST
                                     │
                  ┌──────────────────┼──────────────────┐
                  ▼                  ▼                  ▼
              Zero return          Ridge               v9
            ```

            ### Key Assumptions

            - Features at date **t** only use observations available through **t**.
            - The target is the next session's log return, **t+1**.
            - Feature scaling is fit on the training partition only.
            - Train: targets before 2025; validation: 2025 targets; test: 2026 targets.
            - Validation controls early stopping. Test observations never select model weights.
            - The notebook uses a frozen CSV when present; otherwise it downloads SPY from Yahoo Finance and writes that snapshot.
            - Transaction-cost results are diagnostics, not an execution simulator.
            """
        ),
        code(
            """
            from __future__ import annotations

            import json
            import os
            import random
            import time
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            import yfinance as yf
            from IPython.display import display
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from tensorflow.keras import Model
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Lambda
            from tensorflow.keras.losses import Huber
            from tensorflow.keras.optimizers import Adam

            SEED = 42
            WINDOW = 30
            TRAIN_END = pd.Timestamp("2025-01-01")
            VALIDATION_END = pd.Timestamp("2026-01-01")
            DATA_END = "2026-09-27"
            MAX_EPOCHS = 20
            BATCH_SIZE = 64
            TRANSACTION_COST_BPS = 5.0

            ROOT = Path.cwd().resolve()
            while ROOT.parent != ROOT and not (ROOT / "pyproject.toml").exists():
                ROOT = ROOT.parent
            if not (ROOT / "pyproject.toml").exists():
                raise RuntimeError("Run the notebook from inside the repository")

            DATA_PATH = ROOT / "examples" / "data" / "v9_spy_2015_2026.csv"
            OUTPUT_DIR = ROOT / "reports" / "generated" / "v9-notebook"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

            os.environ["PYTHONHASHSEED"] = str(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            print(f"TensorFlow: {tf.__version__}")
            print(f"Window: {WINDOW} sessions")
            print(f"Train targets: < {TRAIN_END.date()}")
            print(f"Validation targets: 2025")
            print(f"Test targets: 2026")
            """
        ),
        md("## Data\n\n### 1. Load or freeze SPY adjusted close"),
        code(
            """
            def load_close() -> pd.Series:
                if DATA_PATH.exists():
                    frame = pd.read_csv(DATA_PATH, parse_dates=["Date"])
                    close = frame.set_index("Date")["Close"].astype(float)
                    source = f"frozen CSV: {DATA_PATH.relative_to(ROOT)}"
                else:
                    last_error = None
                    for attempt in range(1, 4):
                        try:
                            frame = yf.download(
                                "SPY",
                                start="2015-01-01",
                                end=DATA_END,
                                auto_adjust=True,
                                progress=False,
                                threads=False,
                            )
                            if frame.empty:
                                raise RuntimeError("Yahoo Finance returned no SPY data")
                            close = frame["Close"]
                            if isinstance(close, pd.DataFrame):
                                close = close.iloc[:, 0]
                            close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
                            close.index = pd.to_datetime(close.index).tz_localize(None)
                            close.index.name = "Date"
                            close.rename("Close").to_frame().to_csv(DATA_PATH)
                            source = f"Yahoo Finance; frozen to {DATA_PATH.relative_to(ROOT)}"
                            break
                        except Exception as exc:
                            last_error = exc
                            if attempt == 3:
                                raise
                            time.sleep(5 * attempt)
                    else:
                        raise RuntimeError(last_error)
                close = close.sort_index()
                close.name = "Close"
                print(source)
                print(f"{close.index.min().date()} → {close.index.max().date()} | {len(close):,} observations")
                return close

            close = load_close()
            display(close.tail().to_frame())
            """
        ),
        md("### 2. Build stationary features and next-session target"),
        code(
            """
            features = pd.DataFrame(index=close.index)
            features["log_return_pct"] = 100.0 * np.log(close / close.shift(1))
            features["vol_5_pct"] = features["log_return_pct"].rolling(5).std()
            features["vol_20_pct"] = features["log_return_pct"].rolling(20).std()
            features["momentum_5_pct"] = 100.0 * np.log(close / close.shift(5))
            features["momentum_20_pct"] = 100.0 * np.log(close / close.shift(20))
            features["distance_sma20_pct"] = 100.0 * (close / close.rolling(20).mean() - 1.0)

            research = features.copy()
            research["target_return_pct"] = features["log_return_pct"].shift(-1)
            research["target_date"] = research.index.to_series().shift(-1)
            research["target_close"] = close.shift(-1)
            research["current_close"] = close
            research = research.dropna().copy()

            FEATURE_COLUMNS = list(features.columns)
            print("Features:", FEATURE_COLUMNS)
            display(research.tail(5))
            """
        ),
        md("### 3. Fit preprocessing on training rows only"),
        code(
            """
            train_row_mask = research["target_date"] < TRAIN_END
            feature_scaler = StandardScaler()
            feature_scaler.fit(research.loc[train_row_mask, FEATURE_COLUMNS])

            scaled_features = pd.DataFrame(
                feature_scaler.transform(research[FEATURE_COLUMNS]),
                index=research.index,
                columns=FEATURE_COLUMNS,
            )

            def make_sequences():
                x, y_return, y_direction = [], [], []
                target_dates, current_close_values, target_close_values = [], [], []
                for i in range(WINDOW - 1, len(research)):
                    x.append(
                        scaled_features.iloc[i - WINDOW + 1 : i + 1].to_numpy(dtype=np.float32)
                    )
                    target_return = float(research.iloc[i]["target_return_pct"])
                    y_return.append(target_return)
                    y_direction.append(float(target_return > 0.0))
                    target_dates.append(pd.Timestamp(research.iloc[i]["target_date"]))
                    current_close_values.append(float(research.iloc[i]["current_close"]))
                    target_close_values.append(float(research.iloc[i]["target_close"]))
                return (
                    np.asarray(x, dtype=np.float32),
                    np.asarray(y_return, dtype=np.float32),
                    np.asarray(y_direction, dtype=np.float32),
                    pd.DatetimeIndex(target_dates),
                    np.asarray(current_close_values, dtype=float),
                    np.asarray(target_close_values, dtype=float),
                )

            x, y_return, y_direction, target_dates, current_close_values, target_close_values = make_sequences()

            train_mask = target_dates < TRAIN_END
            validation_mask = (target_dates >= TRAIN_END) & (target_dates < VALIDATION_END)
            test_mask = target_dates >= VALIDATION_END

            x_train, y_train, d_train = x[train_mask], y_return[train_mask], y_direction[train_mask]
            x_val, y_val, d_val = x[validation_mask], y_return[validation_mask], y_direction[validation_mask]
            x_test, y_test, d_test = x[test_mask], y_return[test_mask], y_direction[test_mask]

            test_dates = target_dates[test_mask]
            test_current_close = current_close_values[test_mask]
            test_target_close = target_close_values[test_mask]

            split_table = pd.DataFrame({
                "split": ["train", "validation", "test"],
                "samples": [len(x_train), len(x_val), len(x_test)],
                "first_target": [
                    target_dates[train_mask].min(),
                    target_dates[validation_mask].min(),
                    target_dates[test_mask].min(),
                ],
                "last_target": [
                    target_dates[train_mask].max(),
                    target_dates[validation_mask].max(),
                    target_dates[test_mask].max(),
                ],
            })
            display(split_table)
            """
        ),
        md("## Results\n\n### 4. Baselines: zero return and Ridge"),
        code(
            """
            zero_prediction = np.zeros_like(y_test)

            ridge = Ridge(alpha=10.0)
            ridge.fit(x_train.reshape(len(x_train), -1), y_train)
            ridge_prediction = ridge.predict(x_test.reshape(len(x_test), -1)).astype(float)

            print(f"Ridge coefficients: {ridge.coef_.size:,}")
            print(f"Ridge train R²: {ridge.score(x_train.reshape(len(x_train), -1), y_train):.4f}")
            """
        ),
        md("### 5. Build v9 multi-task LSTM"),
        code(
            """
            QUANTILES = tf.constant([0.10, 0.50, 0.90], dtype=tf.float32)

            def pinball_loss(y_true, y_pred):
                y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
                error = y_true - y_pred
                return tf.reduce_mean(
                    tf.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
                )

            def build_v9(input_shape) -> Model:
                inputs = Input(shape=input_shape, name="market_window")
                shared = LSTM(64, return_sequences=True, name="shared_lstm_1")(inputs)
                shared = Dropout(0.10, name="shared_dropout_1")(shared)
                shared = LSTM(32, name="shared_lstm_2")(shared)
                shared = Dropout(0.10, name="shared_dropout_2")(shared)

                direction = Dense(1, activation="sigmoid", name="direction")(shared)
                expected_return = Dense(1, activation="linear", name="expected_return")(shared)
                raw_quantiles = Dense(3, activation="linear", name="raw_quantiles")(shared)
                quantiles = Lambda(
                    lambda values: tf.sort(values, axis=-1),
                    name="quantiles",
                )(raw_quantiles)

                return Model(
                    inputs=inputs,
                    outputs={
                        "direction": direction,
                        "expected_return": expected_return,
                        "quantiles": quantiles,
                    },
                    name="return_multitask_lstm_v9",
                )

            tf.keras.backend.clear_session()
            tf.random.set_seed(SEED)
            v9 = build_v9((x_train.shape[1], x_train.shape[2]))
            v9.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss={
                    "direction": "binary_crossentropy",
                    "expected_return": Huber(),
                    "quantiles": pinball_loss,
                },
                loss_weights={
                    "direction": 0.25,
                    "expected_return": 1.0,
                    "quantiles": 0.50,
                },
                metrics={
                    "direction": ["accuracy"],
                    "expected_return": ["mae"],
                    "quantiles": [],
                },
            )
            v9.summary()
            """
        ),
        md("### 6. Train chronologically with 2025 validation"),
        code(
            """
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2,
                    min_lr=1e-5,
                    verbose=1,
                ),
            ]

            history = v9.fit(
                x_train,
                {
                    "direction": d_train,
                    "expected_return": y_train,
                    "quantiles": y_train,
                },
                validation_data=(
                    x_val,
                    {
                        "direction": d_val,
                        "expected_return": y_val,
                        "quantiles": y_val,
                    },
                ),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=False,
                callbacks=callbacks,
                verbose=0,
            )

            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            print(f"Epochs executed: {len(history.history['loss'])}")
            print(f"Best validation-loss epoch: {best_epoch}")
            print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
            """
        ),
        md("### 7. Held-out 2026 predictions"),
        code(
            """
            v9_output = v9.predict(x_test, verbose=0)
            v9_prediction = np.asarray(v9_output["expected_return"]).reshape(-1)
            v9_direction_probability = np.asarray(v9_output["direction"]).reshape(-1)
            v9_quantiles = np.asarray(v9_output["quantiles"])

            def information_coefficient(actual, predicted):
                if np.std(actual) == 0 or np.std(predicted) == 0:
                    return np.nan
                return float(np.corrcoef(actual, predicted)[0, 1])

            def evaluate_forecast(name, prediction):
                error = np.asarray(prediction) - y_test
                rmse = float(np.sqrt(np.mean(error ** 2)))
                mae = float(np.mean(np.abs(error)))
                direction_accuracy = float(
                    np.mean((np.asarray(prediction) > 0) == (y_test > 0))
                )
                ic = information_coefficient(y_test, prediction)
                zero_rmse = float(np.sqrt(np.mean(y_test ** 2)))
                skill = 1.0 - rmse / zero_rmse
                return {
                    "model": name,
                    "return_rmse_pct": rmse,
                    "return_mae_pct": mae,
                    "direction_accuracy": direction_accuracy,
                    "delta_ic": ic,
                    "rmse_skill_vs_zero": skill,
                }

            summary = pd.DataFrame([
                evaluate_forecast("Zero return", zero_prediction),
                evaluate_forecast("Ridge", ridge_prediction),
                evaluate_forecast("v9 multi-task LSTM", v9_prediction),
            ])

            coverage_80 = float(
                np.mean((y_test >= v9_quantiles[:, 0]) & (y_test <= v9_quantiles[:, 2]))
            )
            mean_interval_width = float(
                np.mean(v9_quantiles[:, 2] - v9_quantiles[:, 0])
            )
            direction_head_accuracy = float(
                np.mean((v9_direction_probability >= 0.5) == (y_test > 0))
            )

            display(summary.round(6))
            print()
            for row in summary.to_dict(orient="records"):
                print(
                    f"{row['model']:<20} "
                    f"RMSE={row['return_rmse_pct']:.4f}%  "
                    f"skill={row['rmse_skill_vs_zero']:+.2%}  "
                    f"direction={row['direction_accuracy']:.2%}  "
                    f"IC={row['delta_ic']:.4f}"
                )
            print(f"v9 direction-head accuracy: {direction_head_accuracy:.2%}")
            print(f"Nominal 80% quantile interval coverage: {coverage_80:.2%}")
            print(f"Mean q10-q90 interval width: {mean_interval_width:.3f} percentage points")
            """
        ),
        md("### 8. Architecture diagram"),
        code(
            """
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.axis("off")

            boxes = {
                "features": (0.04, 0.39, 0.20, 0.22, "30-session window\\nreturns · vol · momentum"),
                "lstm1": (0.31, 0.52, 0.16, 0.16, "LSTM 64"),
                "lstm2": (0.31, 0.25, 0.16, 0.16, "LSTM 32"),
                "direction": (0.57, 0.67, 0.18, 0.15, "Direction head\\nP(up)"),
                "return": (0.57, 0.42, 0.18, 0.15, "Return head\\nE[r(t+1)]"),
                "quantile": (0.57, 0.17, 0.18, 0.15, "Quantile head\\nq10 · q50 · q90"),
                "eval": (0.81, 0.39, 0.16, 0.22, "2026 held-out\\nbenchmark"),
            }

            for x0, y0, width, height, label in boxes.values():
                rect = plt.Rectangle((x0, y0), width, height, fill=False, linewidth=1.8)
                ax.add_patch(rect)
                ax.text(
                    x0 + width / 2,
                    y0 + height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=11,
                )

            def arrow(start, end):
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={"arrowstyle": "->", "lw": 1.5},
                )

            arrow((0.24, 0.50), (0.31, 0.60))
            arrow((0.39, 0.52), (0.39, 0.41))
            arrow((0.47, 0.33), (0.57, 0.745))
            arrow((0.47, 0.33), (0.57, 0.495))
            arrow((0.47, 0.33), (0.57, 0.245))
            arrow((0.75, 0.745), (0.81, 0.55))
            arrow((0.75, 0.495), (0.81, 0.50))
            arrow((0.75, 0.245), (0.81, 0.45))

            ax.set_title("v9 return-first multi-task architecture", fontsize=15, pad=15)
            plt.show()
            """
        ),
        md("### 9. Forecast quality on the held-out period"),
        code(
            """
            plot_frame = pd.DataFrame(
                {
                    "Actual next-session return": y_test,
                    "Ridge": ridge_prediction,
                    "v9": v9_prediction,
                    "q10": v9_quantiles[:, 0],
                    "q90": v9_quantiles[:, 2],
                },
                index=test_dates,
            )
            recent = plot_frame.tail(80)

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(
                recent.index,
                recent["Actual next-session return"],
                label="Actual",
                linewidth=1.2,
            )
            ax.plot(
                recent.index,
                recent["v9"],
                label="v9 expected return",
                linewidth=1.2,
            )
            ax.fill_between(
                recent.index,
                recent["q10"],
                recent["q90"],
                alpha=0.18,
                label="v9 q10-q90",
            )
            ax.axhline(0.0, linewidth=0.8)
            ax.set_title("SPY — last 80 held-out next-session return forecasts")
            ax.set_ylabel("log return (%)")
            ax.legend()
            plt.xticks(rotation=25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md("### 10. Cost-aware directional diagnostic"),
        code(
            """
            actual_simple_returns = np.expm1(y_test / 100.0)
            cost_rate = TRANSACTION_COST_BPS / 10_000.0

            def strategy_series(prediction):
                position = np.where(np.asarray(prediction) > 0, 1.0, -1.0)
                previous_position = np.r_[0.0, position[:-1]]
                turnover = np.abs(position - previous_position)
                strategy_return = position * actual_simple_returns - cost_rate * turnover
                cumulative = np.cumprod(1.0 + strategy_return) - 1.0
                annualized_vol = float(np.std(strategy_return, ddof=1) * np.sqrt(252))
                annualized_return = float(np.mean(strategy_return) * 252)
                sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0
                peak = np.maximum.accumulate(1.0 + cumulative)
                drawdown = (1.0 + cumulative) / peak - 1.0
                return (
                    position,
                    strategy_return,
                    cumulative,
                    sharpe,
                    float(drawdown.min()),
                    float(np.mean(turnover)),
                )

            ridge_strategy = strategy_series(ridge_prediction)
            v9_strategy = strategy_series(v9_prediction)
            buy_hold_cumulative = np.cumprod(1.0 + actual_simple_returns) - 1.0

            strategy_summary = pd.DataFrame([
                {
                    "strategy": "Ridge sign",
                    "sharpe": ridge_strategy[3],
                    "max_drawdown": ridge_strategy[4],
                    "average_turnover": ridge_strategy[5],
                    "final_return": ridge_strategy[2][-1],
                },
                {
                    "strategy": "v9 sign",
                    "sharpe": v9_strategy[3],
                    "max_drawdown": v9_strategy[4],
                    "average_turnover": v9_strategy[5],
                    "final_return": v9_strategy[2][-1],
                },
            ])
            display(strategy_summary.round(6))

            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(test_dates, buy_hold_cumulative, label="SPY buy & hold")
            ax.plot(test_dates, ridge_strategy[2], label="Ridge sign, after 5 bps")
            ax.plot(test_dates, v9_strategy[2], label="v9 sign, after 5 bps")
            ax.set_title("2026 held-out cumulative-return diagnostic")
            ax.set_ylabel("cumulative return")
            ax.legend()
            plt.xticks(rotation=25)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Takeaways

            The cells above are intentionally evidence-first:

            - **Return RMSE skill vs zero** answers whether the model improves on "tomorrow's return = 0".
            - **Ridge** checks whether the neural network is adding anything beyond a simple linear model on the same information.
            - **Direction accuracy and IC** test whether the sign/ranking signal is useful.
            - **q10-q90 coverage** tests whether the uncertainty head behaves plausibly.
            - The cost-aware strategy is shown only as a diagnostic; it is not evidence of deployable trading performance by itself.

            An unimpressive result is still useful: it tells us to improve the target/features/model rather than rely on a visually convincing price chart.
            """
        ),
        code(
            """
            result_payload = {
                "seed": SEED,
                "window": WINDOW,
                "train_samples": int(len(x_train)),
                "validation_samples": int(len(x_val)),
                "test_samples": int(len(x_test)),
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                "best_epoch": best_epoch,
                "forecast_metrics": summary.to_dict(orient="records"),
                "direction_head_accuracy": direction_head_accuracy,
                "interval_80_coverage": coverage_80,
                "interval_mean_width_pct": mean_interval_width,
                "ridge_strategy": {
                    "sharpe": ridge_strategy[3],
                    "max_drawdown": ridge_strategy[4],
                    "average_turnover": ridge_strategy[5],
                    "final_return": float(ridge_strategy[2][-1]),
                },
                "v9_strategy": {
                    "sharpe": v9_strategy[3],
                    "max_drawdown": v9_strategy[4],
                    "average_turnover": v9_strategy[5],
                    "final_return": float(v9_strategy[2][-1]),
                },
            }
            metrics_path = OUTPUT_DIR / "v9_spy_metrics.json"
            metrics_path.write_text(
                json.dumps(result_payload, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(result_payload, indent=2))
            """
        ),
    ]
    return nb


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "examples" / "notebooks" / "v9_return_forecasting.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        nbf.write(build_notebook(), handle)
    print(output)


if __name__ == "__main__":
    main()
