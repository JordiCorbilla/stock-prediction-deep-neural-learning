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
import json
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from datetime import datetime, timedelta

import exchange_calendars as xcals
import tensorflow as tf
from pandas.tseries.offsets import BDay

from quant_forecast_lab.experiment import sha256_file
from quant_forecast_lab.legacy_keras import load_legacy_h5
from quant_forecast_lab.uncertainty import symmetric_conformal_interval
from stock_prediction_class import StockPrediction
from stock_prediction_numpy import StockData


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
    with open(config_path, encoding='utf-8') as config_file:
        return json.load(config_file)


def _future_dates(last_date, forecast_days, use_business_days, exchange_calendar=None):
    if not isinstance(forecast_days, int) or forecast_days < 1:
        raise ValueError('forecast_days must be a positive integer.')
    if exchange_calendar:
        calendar = xcals.get_calendar(exchange_calendar)
        first_session = calendar.date_to_session(
            pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1),
            direction='next',
        )
        return calendar.sessions_window(first_session, forecast_days)
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


def _return_model_weight(step_index, half_life_days):
    """Weight given to recursively generated model signal at a future horizon."""
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    return float(0.5 ** (float(step_index) / float(half_life_days)))


def _anchor_return(raw_return, anchor_return, step_index, half_life_days):
    """Decay recursive return forecasts toward an unconditional return anchor."""
    model_weight = _return_model_weight(step_index, half_life_days)
    anchored = (model_weight * float(raw_return)) + ((1.0 - model_weight) * float(anchor_return))
    return anchored, model_weight


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


def _scale_scalar(scaler, value, *, inverse=False):
    column = str(scaler.feature_names_in_[0]) if hasattr(scaler, 'feature_names_in_') else 'Close'
    sample = pd.DataFrame({column: [value]}) if hasattr(scaler, 'feature_names_in_') else [[value]]
    result = scaler.inverse_transform(sample) if inverse else scaler.transform(sample)
    return float(result[0][0])


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


def _estimate_noise_std(close_series, use_returns, use_deltas, use_trend_residual, lookback):
    if use_returns:
        base_series = np.log(close_series).diff().dropna()
    else:
        base_series = close_series.diff().dropna()
    if lookback and lookback > 0:
        base_series = base_series.tail(lookback)
    if len(base_series) < 2:
        return 0.0
    return float(base_series.std())


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
        stochastic_paths,
        stochastic_seed,
        stochastic_sigma_mult,
        stochastic_lookback,
        conformal_coverage=0.90,
        exchange_calendar=None,
        return_anchor_enabled=True,
        return_anchor_half_life=5.0,
        return_anchor_mode='training_mean',
        output_folder=None,
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
        self.stochastic_paths = stochastic_paths
        self.stochastic_seed = stochastic_seed
        self.stochastic_sigma_mult = stochastic_sigma_mult
        self.stochastic_lookback = stochastic_lookback
        self.conformal_coverage = conformal_coverage
        self.exchange_calendar = exchange_calendar
        self.return_anchor_enabled = return_anchor_enabled
        self.return_anchor_half_life = return_anchor_half_life
        self.return_anchor_mode = return_anchor_mode
        self.output_folder = output_folder

    def run(self):
        print(tf.version.VERSION)
        inference_folder = os.path.join(os.getcwd(), self.run_folder)
        output_folder = os.path.abspath(self.output_folder) if self.output_folder is not None else inference_folder
        os.makedirs(output_folder, exist_ok=True)
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

        return_anchor_value = 0.0
        if model_version == 'v9' and use_returns and self.return_anchor_enabled:
            if self.return_anchor_mode == 'zero':
                return_anchor_value = 0.0
            elif self.return_anchor_mode == 'training_mean':
                if config is not None and config.get('return_anchor_mean') is not None:
                    return_anchor_value = float(config['return_anchor_mean'])
                else:
                    historical_returns = np.log(close_series).diff().dropna()
                    return_anchor_value = float(historical_returns.mean()) if len(historical_returns) else 0.0
                    print('Warning: v9 return anchor missing from model_config.json; using available historical mean.')
            else:
                raise ValueError(
                    "return_anchor_mode must be 'training_mean' or 'zero'."
                )
            print(
                'v9 return anchor: '
                + f'{return_anchor_value * 100:.4f}% daily log return, '
                + f'half-life={self.return_anchor_half_life:g} sessions'
            )

        if model_version == 'v7':
            model_dir_path = os.path.join(inference_folder, 'model_direction.keras')
            model_mag_path = os.path.join(inference_folder, 'model_magnitude.keras')
            dir_model = tf.keras.models.load_model(model_dir_path, compile=False)
            mag_model = tf.keras.models.load_model(model_mag_path, compile=False)
            dir_model.summary()
            model_time_steps = dir_model.input_shape[1]
        elif model_version == 'v8':
            multitask_model_path = os.path.join(inference_folder, 'model_multitask.keras')
            multitask_model = tf.keras.models.load_model(multitask_model_path, compile=False)
            multitask_model.summary()
            model_time_steps = multitask_model.input_shape[1]
        elif model_version == 'v9':
            return_model_path = os.path.join(inference_folder, 'model_return_multitask.keras')
            return_model = tf.keras.models.load_model(return_model_path, compile=False)
            return_model.summary()
            model_time_steps = return_model.input_shape[1]
        else:
            model_path = os.path.join(inference_folder, 'model.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(inference_folder, 'model_weights.h5')
            model = (load_legacy_h5(model_path) if model_path.endswith('.h5')
                     else tf.keras.models.load_model(model_path, compile=False))
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
                raise FileNotFoundError('input_scaler.pkl is required for reproducible inference. Re-export the training scaler with the model.')

        if scaler is None:
            raise FileNotFoundError('min_max_scaler.pkl is required for reproducible inference. Re-export the training scaler with the model.')

        if use_deltas or use_trend_residual:
            window_scaled = _scale_input(input_scaler, recent_window)
        else:
            window_scaled = _scale_input(scaler, recent_window)
        window_scaled = window_scaled.reshape(1, time_steps, 1)
        unanchored_window_scaled = window_scaled.copy() if model_version == 'v9' else None

        future_dates = _future_dates(
            latest_date,
            self.forecast_days,
            self.use_business_days,
            self.exchange_calendar,
        )
        predictions = []
        raw_return_predictions = []
        conditional_return_predictions = []
        return_anchor_weights = []
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
                mag_value = _scale_scalar(scaler, mag_scaled, inverse=True)
                if mag_clip_value is not None:
                    mag_value = min(mag_value, mag_clip_value)
                pred_values = [mag_value if dir_prob >= self.direction_threshold else -mag_value]
                pred_scaled = [mag_scaled]
            elif model_version == 'v8':
                multitask_pred = multitask_model.predict(window_scaled, verbose=0)
                dir_prob = float(multitask_pred['direction'][0][0])
                mag_scaled = float(multitask_pred['magnitude'][0][0])
                mag_value = _scale_scalar(scaler, mag_scaled, inverse=True)
                if mag_clip_value is not None:
                    mag_value = min(mag_value, mag_clip_value)
                pred_values = [mag_value if dir_prob >= self.direction_threshold else -mag_value]
                pred_scaled = [mag_scaled]
            elif model_version == 'v9':
                return_pred = return_model.predict(
                    np.concatenate((window_scaled, unanchored_window_scaled), axis=0), verbose=0
                )
                dir_prob = float(return_pred['direction'][0][0])
                return_scaled_raw = float(return_pred['expected_return'][0][0])
                conditional_return = _scale_scalar(scaler, return_scaled_raw, inverse=True)
                unanchored_scaled = float(return_pred['expected_return'][1][0])
                return_value_raw = _scale_scalar(scaler, unanchored_scaled, inverse=True)

                if self.return_anchor_enabled:
                    return_value, model_weight = _anchor_return(
                        conditional_return,
                        return_anchor_value,
                        step_index,
                        self.return_anchor_half_life,
                    )
                else:
                    return_value = conditional_return
                    model_weight = 1.0

                return_scaled = _scale_scalar(scaler, return_value)
                raw_return_predictions.append(return_value_raw)
                conditional_return_predictions.append(conditional_return)
                return_anchor_weights.append(model_weight)
                pred_values = [return_value]
                pred_scaled = [return_scaled]
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
                    if model_version == 'v9':
                        unanchored_window_scaled = np.concatenate(
                            [unanchored_window_scaled[:, 1:, :], [[[unanchored_scaled]]]], axis=1
                        )
                step_index += 1
            if step_index > 0 and step_index % max(1, steps // 10) == 0:
                print(f'Inference progress: {step_index}/{steps}')

        predicted_prices_unanchored = None
        if use_returns:
            predicted_prices_raw = _returns_to_prices(predictions, latest_close_price)
            if model_version == 'v9' and raw_return_predictions:
                predicted_prices_unanchored = _returns_to_prices(
                    raw_return_predictions,
                    latest_close_price,
                )
        elif use_deltas or use_trend_residual:
            predicted_prices_raw = latest_close_price + np.cumsum(predictions)
        else:
            predicted_prices_raw = predictions
            if self.clip_negative:
                predicted_prices_raw = np.maximum(predicted_prices_raw, 0)

        predicted_prices = self._blend_predictions(predicted_prices_raw, latest_close_price)
        stochastic_paths = self._build_stochastic_paths(
            predictions,
            latest_close_price,
            use_returns,
            use_deltas,
            use_trend_residual,
            close_series,
        )

        stochastic_summary = {}
        if stochastic_paths is not None and len(stochastic_paths) > 0:
            p10 = np.percentile(stochastic_paths, 10, axis=0)
            p50 = np.percentile(stochastic_paths, 50, axis=0)
            p90 = np.percentile(stochastic_paths, 90, axis=0)
            stochastic_summary = {
                'Predicted_Price_P10': p10,
                'Predicted_Price_P50': p50,
                'Predicted_Price_P90': p90,
            }

        forecast_df = pd.DataFrame(
            {
                'Date': future_dates,
                'Predicted_Price': predicted_prices,
                'Predicted_Price_Raw': predicted_prices_raw,
                'Predicted_Return': predictions if use_returns else np.nan,
                'Predicted_Return_Raw': (
                    raw_return_predictions
                    if model_version == 'v9' and use_returns
                    else np.nan
                ),
                'Predicted_Return_Conditional': (
                    conditional_return_predictions
                    if model_version == 'v9' and use_returns
                    else np.nan
                ),
                'Return_Anchor': (
                    [return_anchor_value] * len(future_dates)
                    if model_version == 'v9' and use_returns
                    else np.nan
                ),
                'Return_Model_Weight': (
                    return_anchor_weights
                    if model_version == 'v9' and use_returns
                    else np.nan
                ),
                'Predicted_Price_Unanchored': (
                    predicted_prices_unanchored
                    if predicted_prices_unanchored is not None
                    else np.nan
                ),
                'Predicted_Delta': predictions if use_deltas else np.nan,
                'Predicted_Trend_Residual': predictions if use_trend_residual else np.nan,
                **stochastic_summary,
            }
        ).set_index('Date')

        in_sample_full = _load_in_sample_predictions(inference_folder, self.ticker)
        if in_sample_full is not None and len(in_sample_full) >= 10:
            calibration = pd.DataFrame(
                {
                    'actual': close_series.reindex(in_sample_full.index),
                    'predicted': in_sample_full.iloc[:, 0],
                }
            ).dropna()
            if len(calibration) >= 10:
                lower, upper = symmetric_conformal_interval(
                    forecast_df['Predicted_Price'].to_numpy(),
                    calibration['actual'].to_numpy(),
                    calibration['predicted'].to_numpy(),
                    coverage=self.conformal_coverage,
                )
                coverage_pct = int(round(self.conformal_coverage * 100))
                forecast_df[f'Predicted_Price_OneStepResidual{coverage_pct}_Lower'] = lower
                forecast_df[f'Predicted_Price_OneStepResidual{coverage_pct}_Upper'] = upper

        forecast_df.to_csv(os.path.join(output_folder, 'future_predictions.csv'))
        inference_config = {
            'model_version': model_version,
            'model_sha256': (
                {'direction': sha256_file(model_dir_path), 'magnitude': sha256_file(model_mag_path)}
                if model_version == 'v7' else
                sha256_file(multitask_model_path) if model_version == 'v8' else
                sha256_file(return_model_path) if model_version == 'v9' else
                sha256_file(model_path)
            ),
            'last_observed_date': pd.Timestamp(latest_date).strftime('%Y-%m-%d'),
            'last_observed_close': latest_close_price,
            'forecast_days': self.forecast_days,
            'exchange_calendar': self.exchange_calendar,
            'blend_alpha': self.blend_alpha,
            'return_anchor_enabled': self.return_anchor_enabled if model_version == 'v9' else None,
            'return_anchor_mode': self.return_anchor_mode if model_version == 'v9' else None,
            'return_anchor_value': return_anchor_value if model_version == 'v9' else None,
            'return_anchor_half_life': self.return_anchor_half_life if model_version == 'v9' else None,
            'one_step_residual_band_nominal_coverage': self.conformal_coverage,
            'recursive_horizon_coverage_validated': False,
        }
        with open(os.path.join(output_folder, 'inference_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(inference_config, handle, indent=2, allow_nan=False)

        if len(forecast_df) > 0:
            first_pred = float(forecast_df['Predicted_Price'].iloc[0])
            latest_price = float(latest_close_price)
            delta_pct = ((first_pred - latest_price) / latest_price) * 100
            print('Sanity check - next day delta: ' + f'{delta_pct:.2f}%')

        history = close_series.tail(self.plot_history_days)
        if in_sample_full is not None:
            in_sample = in_sample_full.tail(self.plot_history_days)
        else:
            in_sample = None
        plt.figure(figsize=(14, 5))
        plt.plot(history.index, history, color='green', label='Actual [' + self.ticker + '] price')
        if in_sample is not None and not in_sample.empty:
            plt.plot(in_sample.index, in_sample.iloc[:, 0], color='orange', label='Historical held-out [' + self.ticker + '] predicted')
        if stochastic_paths is not None and len(stochastic_paths) > 0:
            max_paths = min(len(stochastic_paths), 20)
            for idx in range(max_paths):
                plt.plot(
                    forecast_df.index,
                    stochastic_paths[idx],
                    color='gray',
                    alpha=0.2,
                    linewidth=1,
                    label='Scenario paths' if idx == 0 else None,
                )
            if {'Predicted_Price_P10', 'Predicted_Price_P90'}.issubset(forecast_df.columns):
                plt.fill_between(
                    forecast_df.index,
                    forecast_df['Predicted_Price_P10'],
                    forecast_df['Predicted_Price_P90'],
                    color='gray',
                    alpha=0.15,
                    label='Scenario P10-P90 band',
                )
        conformal_lower = [column for column in forecast_df.columns if column.startswith('Predicted_Price_OneStepResidual') and column.endswith('_Lower')]
        if conformal_lower:
            lower_col = conformal_lower[0]
            upper_col = lower_col.replace('_Lower', '_Upper')
            plt.fill_between(
                forecast_df.index,
                forecast_df[lower_col],
                forecast_df[upper_col],
                alpha=0.12,
                label=f'One-step residual band (nominal {int(round(self.conformal_coverage * 100))}%)',
            )
        if (
            model_version == 'v9'
            and 'Predicted_Price_Unanchored' in forecast_df.columns
            and forecast_df['Predicted_Price_Unanchored'].notna().any()
        ):
            plt.plot(
                forecast_df.index,
                forecast_df['Predicted_Price_Unanchored'],
                color='firebrick',
                linestyle='--',
                alpha=0.45,
                linewidth=1.2,
                label='Unanchored recursive v9 path',
            )
        plt.plot(
            forecast_df.index,
            forecast_df['Predicted_Price'],
            color='red',
            label='Anchored [' + self.ticker + '] prediction' if model_version == 'v9' else 'Predicted [' + self.ticker + '] price',
        )
        plt.xlabel('Time')
        currency = data.get_stock_currency()
        plt.ylabel(f'Price [{currency}]' if currency else 'Price')
        plt.legend()
        plt.title('Actual vs Predicted Prices')
        plt.savefig(os.path.join(output_folder, self.ticker + '_future_forecast.png'))
        plt.show()
        print('Forecast outputs: ' + os.path.abspath(output_folder))
        return forecast_df

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

    def _build_stochastic_paths(
        self,
        predictions,
        latest_close_price,
        use_returns,
        use_deltas,
        use_trend_residual,
        close_series,
    ):
        if self.stochastic_paths <= 0:
            return None
        noise_std = _estimate_noise_std(
            close_series,
            use_returns,
            use_deltas,
            use_trend_residual,
            self.stochastic_lookback,
        )
        if noise_std <= 0:
            print('Warning: stochastic noise std is 0. Skipping stochastic paths.')
            return None
        noise_std *= max(self.stochastic_sigma_mult, 0.0)
        rng = np.random.default_rng(self.stochastic_seed)
        steps = len(predictions)
        noise = rng.normal(0.0, noise_std, size=(self.stochastic_paths, steps))
        predictions = np.asarray(predictions)
        if use_returns:
            paths = []
            for idx in range(self.stochastic_paths):
                returns = predictions + noise[idx]
                paths.append(self._blend_predictions(_returns_to_prices(returns, latest_close_price), latest_close_price))
            return np.asarray(paths)
        if use_deltas or use_trend_residual:
            paths = []
            for idx in range(self.stochastic_paths):
                deltas = predictions + noise[idx]
                prices = latest_close_price + np.cumsum(deltas)
                paths.append(self._blend_predictions(prices, latest_close_price))
            return np.asarray(paths)
        paths = []
        for idx in range(self.stochastic_paths):
            prices = predictions + noise[idx]
            paths.append(self._blend_predictions(prices, latest_close_price))
        return np.asarray(paths)


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
        stochastic_paths=STOCHASTIC_PATHS,
        stochastic_seed=STOCHASTIC_SEED,
        stochastic_sigma_mult=STOCHASTIC_SIGMA_MULT,
        stochastic_lookback=STOCHASTIC_LOOKBACK,
        conformal_coverage=CONFORMAL_COVERAGE,
        exchange_calendar=EXCHANGE_CALENDAR,
        return_anchor_enabled=RETURN_ANCHOR_ENABLED,
        return_anchor_half_life=RETURN_ANCHOR_HALF_LIFE,
        return_anchor_mode=RETURN_ANCHOR_MODE,
        )
        runner.run()

if __name__ == '__main__':
    TIME_STEPS = 3
    RUN_FOLDER = os.path.join('examples', 'runs', 'reference-v7-ftse')
    TOKEN = 'reference-v7-ftse'
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
    STOCHASTIC_PATHS = 50
    STOCHASTIC_SEED = 42
    STOCHASTIC_SIGMA_MULT = 0.6
    STOCHASTIC_LOOKBACK = 120
    CONFORMAL_COVERAGE = 0.90
    EXCHANGE_CALENDAR = 'XLON'
    RETURN_ANCHOR_ENABLED = True
    RETURN_ANCHOR_HALF_LIFE = 5.0
    RETURN_ANCHOR_MODE = 'training_mean'
    app.run(main)
