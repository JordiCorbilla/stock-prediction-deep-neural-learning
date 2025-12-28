# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import warnings
from absl import app
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

import tensorflow as tf

from stock_prediction_class import StockPrediction
from stock_prediction_numpy import StockData
from datetime import timedelta, datetime
from pandas.tseries.offsets import BDay

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def _load_scaler(inference_folder):
    scaler_path = os.path.join(inference_folder, 'min_max_scaler.pkl')
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)


def _load_input_scaler(inference_folder):
    scaler_path = os.path.join(inference_folder, 'input_scaler.pkl')
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, 'rb') as scaler_file:
        return pickle.load(scaler_file)


def _load_config(inference_folder):
    config_path = os.path.join(inference_folder, 'model_config.json')
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as config_file:
        return json.load(config_file)


def _future_dates(last_date, forecast_days, use_business_days):
    if use_business_days:
        return pd.bdate_range(last_date + BDay(1), periods=forecast_days)
    return pd.date_range(last_date + timedelta(1), periods=forecast_days)


def _returns_to_prices(returns, start_price):
    prices = []
    current_price = start_price
    for value in returns:
        current_price = current_price * np.exp(value)
        prices.append(current_price)
    return prices


def _ensure_frame(series_or_frame):
    if isinstance(series_or_frame, pd.Series):
        return series_or_frame.to_frame()
    return series_or_frame


def _scale_input(scaler, data):
    if hasattr(scaler, 'feature_names_in_'):
        return scaler.transform(_ensure_frame(data))
    array_data = np.asarray(data)
    if array_data.ndim == 1:
        array_data = array_data.reshape(-1, 1)
    return scaler.transform(array_data)


def _get_close_series(raw_data):
    close_data = raw_data['Close']
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]
    return close_data


def _load_in_sample_predictions(inference_folder, ticker):
    predictions_path = os.path.join(inference_folder, 'predictions.csv')
    if not os.path.exists(predictions_path):
        return None
    predictions = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    if predictions.empty:
        return None
    expected_col = ticker + '_predicted'
    if expected_col in predictions.columns:
        series = predictions[[expected_col]]
    else:
        series = predictions.iloc[:, [0]]
    series.index = pd.to_datetime(series.index, errors='coerce')
    series = series.dropna()
    return series


class InferenceRunner:
    def __init__(
        self,
        run_folder,
        ticker,
        start_date,
        validation_date,
        github_url,
        epochs,
        time_steps,
        token,
        batch_size,
        forecast_days,
        use_business_days,
        plot_history_days,
        use_returns,
        use_deltas,
        clip_negative,
        blend_alpha,
        direction_threshold,
        mag_clip_pct,
    ):
        self.run_folder = run_folder
        self.ticker = ticker
        self.start_date = start_date
        self.validation_date = validation_date
        self.github_url = github_url
        self.epochs = epochs
        self.time_steps = time_steps
        self.token = token
        self.batch_size = batch_size
        self.forecast_days = forecast_days
        self.use_business_days = use_business_days
        self.plot_history_days = plot_history_days
        self.use_returns = use_returns
        self.use_deltas = use_deltas
        self.clip_negative = clip_negative
        self.blend_alpha = blend_alpha
        self.direction_threshold = direction_threshold
        self.mag_clip_pct = mag_clip_pct

    def run(self):
        print(tf.version.VERSION)
        inference_folder = os.path.join(os.getcwd(), self.run_folder)
        stock = StockPrediction(
            self.ticker,
            self.start_date,
            self.validation_date,
            inference_folder,
            self.github_url,
            self.epochs,
            self.time_steps,
            self.token,
            self.batch_size,
        )

        data = StockData(stock)

        raw_data = data.download_raw_data()
        if raw_data.empty:
            print("Error: No data available for inference.")
            return

        close_series = _get_close_series(raw_data)
        print('Latest Stock Price')
        latest_close_price = float(close_series.iloc[-1])
        latest_date = raw_data.index[-1]
        print(latest_close_price)
        print('Latest Date')
        print(latest_date)

        scaler = _load_scaler(inference_folder)
        config = _load_config(inference_folder)
        use_returns = self.use_returns
        use_deltas = self.use_deltas
        use_trend_residual = False
        model_version = 'v1'
        forecast_horizon = 1
        trend_window = 60
        if config is not None:
            use_returns = bool(config.get('use_returns', use_returns))
            use_deltas = bool(config.get('use_deltas', use_deltas))
            use_trend_residual = bool(config.get('use_trend_residual', use_trend_residual))
            model_version = config.get('model_version', model_version)
            forecast_horizon = int(config.get('forecast_horizon', forecast_horizon))
            trend_window = int(config.get('trend_window', trend_window))
            if use_returns != self.use_returns:
                print('Warning: USE_RETURNS overridden by model_config.json')

        if model_version == 'v7':
            model_dir_path = os.path.join(inference_folder, 'model_direction.keras')
            model_mag_path = os.path.join(inference_folder, 'model_magnitude.keras')
            dir_model = tf.keras.models.load_model(model_dir_path, compile=False)
            mag_model = tf.keras.models.load_model(model_mag_path, compile=False)
            dir_model.summary()
            model_time_steps = dir_model.input_shape[1]
        else:
            model_path = os.path.join(inference_folder, 'model.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(inference_folder, 'model_weights.h5')
            model = tf.keras.models.load_model(model_path, compile=False)
            model.summary()
            model_time_steps = model.input_shape[1]

        if model_time_steps and model_time_steps != self.time_steps:
            print('Warning: TIME_STEPS does not match model input. Using model value: ' + str(model_time_steps))
            time_steps = model_time_steps
        else:
            time_steps = self.time_steps

        if use_returns:
            series = np.log(close_series).diff().dropna().rename('Close')
            recent_window = series.tail(time_steps)
        else:
            recent_window = close_series.tail(time_steps).to_frame()

        if len(recent_window) < time_steps:
            print("Error: Not enough data to build the inference window.")
            return

        input_scaler = None
        if use_deltas or use_trend_residual:
            input_scaler = _load_input_scaler(inference_folder)
            if input_scaler is None:
                print('Warning: input_scaler.pkl not found. Fitting input scaler on full dataset for inference.')
                input_scaler = data.get_input_scaler()
                input_scaler.fit(close_series.to_frame())

        if scaler is None:
            print('Warning: min_max_scaler.pkl not found. Fitting scaler on full dataset for inference.')
            scaler = data.get_min_max()
            if use_returns:
                scaler.fit(series.to_frame())
            elif use_deltas:
                deltas = close_series.diff().dropna().rename('Close')
                scaler.fit(deltas.to_frame())
            elif use_trend_residual:
                residuals = data._compute_trend_residuals(close_series, trend_window).rename('Close')
                scaler.fit(residuals.to_frame())
            else:
                scaler.fit(close_series.to_frame())

        if use_deltas or use_trend_residual:
            window_scaled = _scale_input(input_scaler, recent_window)
        else:
            window_scaled = _scale_input(scaler, recent_window)
        window_scaled = window_scaled.reshape(1, time_steps, 1)

        future_dates = _future_dates(latest_date, self.forecast_days, self.use_business_days)
        predictions = []
        current_close = latest_close_price

        steps = len(future_dates)
        step_index = 0
        recent_delta_abs = close_series.diff().abs().dropna()
        if len(recent_delta_abs) > 0 and self.mag_clip_pct > 0:
            mag_clip_value = np.percentile(recent_delta_abs.to_numpy(), self.mag_clip_pct)
        else:
            mag_clip_value = None

        while step_index < steps:
            if step_index % max(1, steps // 10) == 0:
                print(f'Inference progress: {step_index}/{steps}')
            if model_version == 'v7':
                dir_prob = dir_model.predict(window_scaled, verbose=0)[0][0]
                mag_scaled = mag_model.predict(window_scaled, verbose=0)[0][0]
                mag_value = scaler.inverse_transform([[mag_scaled]])[0][0]
                if mag_clip_value is not None:
                    mag_value = min(mag_value, mag_clip_value)
                pred_values = [mag_value if dir_prob >= self.direction_threshold else -mag_value]
                pred_scaled = [mag_scaled]
            else:
                pred_scaled = model.predict(window_scaled, verbose=0)[0]
                if model_version in ('v5', 'v6'):
                    pred_scaled = pred_scaled[:forecast_horizon]
                else:
                    pred_scaled = [pred_scaled[0]]
                pred_values = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
            if step_index == 0:
                print(f'Predicted batch size: {len(pred_values)}')
            if len(pred_values) == 0:
                print('Error: model returned empty prediction batch. Check model_version and forecast_horizon.')
                return
            for idx, pred_value in enumerate(pred_values):
                if step_index >= steps:
                    break
                predictions.append(pred_value)
                if use_deltas or use_trend_residual:
                    current_close = current_close + pred_value
                    next_scaled = _scale_input(input_scaler, pd.DataFrame({'Close': [current_close]}))
                    window_scaled = np.concatenate([window_scaled[:, 1:, :], next_scaled.reshape(1, 1, 1)], axis=1)
                else:
                    pred_scaled_value = float(pred_scaled[idx])
                    window_scaled = np.concatenate([window_scaled[:, 1:, :], [[[pred_scaled_value]]]], axis=1)
                step_index += 1
            if step_index > 0 and step_index % max(1, steps // 10) == 0:
                print(f'Inference progress: {step_index}/{steps}')

        if use_returns:
            predicted_prices_raw = _returns_to_prices(predictions, latest_close_price)
        elif use_deltas or use_trend_residual:
            predicted_prices_raw = latest_close_price + np.cumsum(predictions)
        else:
            predicted_prices_raw = predictions
            if self.clip_negative:
                predicted_prices_raw = np.maximum(predicted_prices_raw, 0)

        predicted_prices = self._blend_predictions(predicted_prices_raw, latest_close_price)

        forecast_df = pd.DataFrame(
            {
                'Date': future_dates,
                'Predicted_Price': predicted_prices,
                'Predicted_Price_Raw': predicted_prices_raw,
                'Predicted_Return': predictions if use_returns else np.nan,
                'Predicted_Delta': predictions if use_deltas else np.nan,
                'Predicted_Trend_Residual': predictions if use_trend_residual else np.nan,
            }
        ).set_index('Date')
        forecast_df.to_csv(os.path.join(inference_folder, 'future_predictions.csv'))

        if len(forecast_df) > 0:
            first_pred = float(forecast_df['Predicted_Price'].iloc[0])
            latest_price = float(latest_close_price)
            delta_pct = ((first_pred - latest_price) / latest_price) * 100
            print('Sanity check - next day delta: ' + f'{delta_pct:.2f}%')

        history = close_series.tail(self.plot_history_days)
        in_sample = _load_in_sample_predictions(inference_folder, self.ticker)
        plt.figure(figsize=(14, 5))
        plt.plot(history.index, history, color='green', label='Actual [' + self.ticker + '] price')
        if in_sample is not None and not in_sample.empty:
            plt.plot(in_sample.index, in_sample.iloc[:, 0], color='orange', label='In-sample [' + self.ticker + '] predicted')
        plt.plot(forecast_df.index, forecast_df['Predicted_Price'], color='red', label='Predicted [' + self.ticker + '] price')
        plt.xlabel('Time')
        plt.ylabel('Price [USD]')
        plt.legend()
        plt.title('Actual vs Predicted Prices')
        plt.savefig(os.path.join(inference_folder, self.ticker + '_future_forecast.png'))
        plt.show()

    def _blend_predictions(self, predictions, anchor_price):
        if self.blend_alpha >= 1.0:
            return np.asarray(predictions)
        blended = []
        current = anchor_price
        for value in predictions:
            blended_value = (self.blend_alpha * value) + ((1.0 - self.blend_alpha) * current)
            blended.append(blended_value)
            current = blended_value
        return np.asarray(blended)


def main(argv):
        runner = InferenceRunner(
        run_folder=RUN_FOLDER,
        ticker=STOCK_TICKER,
        start_date=STOCK_START_DATE,
        validation_date=STOCK_VALIDATION_DATE,
        github_url=GITHUB_URL,
        epochs=EPOCHS,
        time_steps=TIME_STEPS,
        token=TOKEN,
        batch_size=BATCH_SIZE,
        forecast_days=FORECAST_DAYS,
        use_business_days=USE_BUSINESS_DAYS,
        plot_history_days=PLOT_HISTORY_DAYS,
        use_returns=USE_RETURNS,
        use_deltas=USE_DELTAS,
        clip_negative=CLIP_NEGATIVE,
        blend_alpha=BLEND_ALPHA,
        direction_threshold=DIRECTION_THRESHOLD,
        mag_clip_pct=MAG_CLIP_PCT,
        )
        runner.run()

if __name__ == '__main__':
    TIME_STEPS = 3
    RUN_FOLDER = '^FTSE_20240103_edae6b8f5fc742031805151aeba98571'
    TOKEN = 'edae6b8f5fc742031805151aeba98571'
    STOCK_TICKER = '^FTSE'
    BATCH_SIZE = 10
    STOCK_START_DATE = pd.to_datetime('2017-11-01')
    start_date = pd.to_datetime('2017-01-01')
    end_date = datetime.today()
    duration = end_date - start_date
    STOCK_VALIDATION_DATE = start_date + 0.8 * duration
    GITHUB_URL = "https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/"
    EPOCHS = 100
    FORECAST_DAYS = 30
    USE_BUSINESS_DAYS = True
    PLOT_HISTORY_DAYS = 200
    USE_RETURNS = False
    USE_DELTAS = False
    CLIP_NEGATIVE = True
    BLEND_ALPHA = 0.6
    DIRECTION_THRESHOLD = 0.55
    MAG_CLIP_PCT = 90
    app.run(main)
