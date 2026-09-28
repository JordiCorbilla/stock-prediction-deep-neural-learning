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
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def _get_security_info(self):
        try:
            return self._sec.info or {}
        except Exception as exc:
            print('Warning: unable to download ticker metadata:', exc)
            return {}

    def get_stock_short_name(self):
        info = self._get_security_info()
        return info.get('shortName') or info.get('longName') or self._stock.get_ticker()

    def get_min_max(self):
        return self._min_max

    def get_input_scaler(self):
        return self._input_scaler

    def get_stock_currency(self):
        return self._get_security_info().get('currency') or ''

    def _compute_log_returns(self, series):
        return np.log(series).diff().dropna()

    def _compute_deltas(self, series):
        return series.diff().dropna()

    def _compute_trend_residuals(self, series, window):
        values = series.to_numpy()
        residuals = []
        for i in range(1, len(values)):
            start = max(0, i - window)
            y = values[start:i]
            if len(y) < 2:
                residuals.append(values[i] - values[i - 1])
                continue
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            trend_next = slope * len(y) + intercept
            residuals.append(values[i] - trend_next)
        return pd.Series(residuals, index=series.index[1:])

    def _ensure_series(self, series_or_frame):
        if isinstance(series_or_frame, pd.DataFrame):
            return series_or_frame.iloc[:, 0]
        return series_or_frame

    def _ensure_frame(self, series_or_frame):
        if isinstance(series_or_frame, pd.Series):
            return series_or_frame.to_frame()
        return series_or_frame

    @staticmethod
    def _fit_sample_count(total_samples, validation_fraction, purge=0):
        if not 0.0 < validation_fraction < 0.5:
            raise ValueError('validation_fraction must be between 0 and 0.5.')
        if total_samples < 3:
            raise ValueError('At least three training samples are required for a validation split.')
        split = max(1, min(total_samples - 1, int(total_samples * (1.0 - validation_fraction))))
        if purge < 0 or split <= purge:
            raise ValueError('Not enough fit samples after the forecast-horizon purge.')
        return split - purge

    def _download_close_frame(self, end_date):
        raw = yf.download(
            self._stock.get_ticker(),
            start=self._stock.get_start_date(),
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        if raw.empty:
            raise ValueError('No market data returned for ticker ' + self._stock.get_ticker())
        if 'Close' not in raw.columns:
            raise ValueError('Downloaded market data does not contain a Close column.')

        close = raw['Close']
        if isinstance(close, pd.DataFrame):
            ticker = self._stock.get_ticker()
            if ticker in close.columns:
                close = close[ticker]
            else:
                close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors='coerce').dropna()
        frame = close.rename('Close').to_frame()
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = 'Date'
        return frame

    def download_raw_data(self, end_date=None):
        if end_date is None:
            end_date = datetime.today()
        return self._download_close_frame(end_date)

    def download_transform_to_numpy(self, time_steps, project_folder, use_returns=False, use_deltas=False, use_trend_residual=False, trend_window=60, forecast_horizon=1, validation_fraction=0.15):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = self._download_close_frame(end_date).reset_index()
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        #print(data)

        training_data = data[data['Date'] < self._stock.get_validation_date()].copy()
        test_data = data[data['Date'] >= self._stock.get_validation_date()].copy()
        training_data = training_data.set_index('Date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Date')
        #print(test_data)

        if use_returns and use_deltas:
            raise ValueError('use_returns and use_deltas cannot both be true')
        if use_trend_residual and use_returns:
            raise ValueError('use_trend_residual cannot be combined with returns')

        if use_returns:
            full_series = data.set_index('Date')[['Close']]
            full_series = self._ensure_series(full_series)
            returns = self._compute_log_returns(full_series).rename('Close')
            training_returns = returns[returns.index < self._stock.get_validation_date()]
            test_returns = returns[returns.index >= self._stock.get_validation_date()]
            fit_count = self._fit_sample_count(len(training_returns) - time_steps, validation_fraction)
            self._min_max.fit(training_returns.iloc[:time_steps + fit_count].to_frame())
            train_scaled = self._min_max.transform(training_returns.to_frame())
        elif use_deltas:
            full_series = data.set_index('Date')[['Close']]
            full_series = self._ensure_series(full_series)
            deltas = self._compute_deltas(full_series).rename('Close')
            training_deltas = deltas[deltas.index < self._stock.get_validation_date()]
            test_deltas = deltas[deltas.index >= self._stock.get_validation_date()]
            fit_count = self._fit_sample_count(len(training_deltas) - time_steps - forecast_horizon + 1, validation_fraction, forecast_horizon - 1)
            fit_end = time_steps + fit_count + forecast_horizon - 1
            self._input_scaler.fit(training_data.iloc[:time_steps + fit_count])
            self._min_max.fit(training_deltas.iloc[:fit_end].to_frame())
            close_scaled = self._input_scaler.transform(training_data)
            delta_scaled = self._min_max.transform(training_deltas.to_frame())
            train_scaled = close_scaled
        elif use_trend_residual:
            full_series = data.set_index('Date')[['Close']]
            full_series = self._ensure_series(full_series)
            residuals = self._compute_trend_residuals(full_series, trend_window).rename('Close')
            training_residuals = residuals[residuals.index < self._stock.get_validation_date()]
            test_residuals = residuals[residuals.index >= self._stock.get_validation_date()]
            fit_count = self._fit_sample_count(len(training_residuals) - time_steps - forecast_horizon + 1, validation_fraction, forecast_horizon - 1)
            fit_end = time_steps + fit_count + forecast_horizon - 1
            self._input_scaler.fit(training_data.iloc[:time_steps + fit_count])
            self._min_max.fit(training_residuals.iloc[:fit_end].to_frame())
            close_scaled = self._input_scaler.transform(training_data)
            residual_scaled = self._min_max.transform(training_residuals.to_frame())
            train_scaled = close_scaled
        else:
            fit_count = self._fit_sample_count(len(training_data) - time_steps, validation_fraction)
            self._min_max.fit(training_data.iloc[:time_steps + fit_count])
            train_scaled = self._min_max.transform(training_data)
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        if use_deltas:
            close_scaled_aligned = close_scaled[1:]
            for i in range(time_steps, close_scaled_aligned.shape[0] - forecast_horizon + 1):
                x_train.append(close_scaled_aligned[i - time_steps:i])
                y_train.append(delta_scaled[i:i + forecast_horizon, 0])
        elif use_trend_residual:
            close_scaled_aligned = close_scaled[1:]
            for i in range(time_steps, close_scaled_aligned.shape[0] - forecast_horizon + 1):
                x_train.append(close_scaled_aligned[i - time_steps:i])
                y_train.append(residual_scaled[i:i + forecast_horizon, 0])
        else:
            for i in range(time_steps, train_scaled.shape[0]):
                x_train.append(train_scaled[i - time_steps:i])
                y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        if use_returns:
            total_returns = pd.concat((training_returns, test_returns), axis=0)
            inputs = total_returns[len(total_returns) - len(test_returns) - time_steps:]
            test_scaled = self._min_max.transform(inputs.to_frame())
        elif use_deltas:
            total_data = pd.concat((training_data, test_data), axis=0)
            total_close_scaled = self._input_scaler.transform(total_data)
            total_close_aligned = total_close_scaled[1:]

            total_deltas = pd.concat((training_deltas, test_deltas), axis=0)
            total_deltas_scaled = self._min_max.transform(total_deltas.to_frame())

            test_start = len(training_deltas)
            inputs_x = total_close_aligned
            inputs_y = total_deltas_scaled
        elif use_trend_residual:
            total_data = pd.concat((training_data, test_data), axis=0)
            total_close_scaled = self._input_scaler.transform(total_data)
            total_close_aligned = total_close_scaled[1:]

            total_residuals = pd.concat((training_residuals, test_residuals), axis=0)
            total_residuals_scaled = self._min_max.transform(total_residuals.to_frame())

            test_start = len(training_residuals)
            inputs_x = total_close_aligned
            inputs_y = total_residuals_scaled
        else:
            total_data = pd.concat((training_data, test_data), axis=0)
            inputs = total_data[len(total_data) - len(test_data) - time_steps:]
            test_scaled = self._min_max.transform(inputs)

        # Testing Data Transformation
        x_test = []
        y_test = []
        if use_deltas:
            for i in range(test_start, inputs_y.shape[0] - forecast_horizon + 1):
                x_test.append(inputs_x[i - time_steps:i])
                y_test.append(inputs_y[i:i + forecast_horizon, 0])
        elif use_trend_residual:
            for i in range(test_start, inputs_y.shape[0] - forecast_horizon + 1):
                x_test.append(inputs_x[i - time_steps:i])
                y_test.append(inputs_y[i:i + forecast_horizon, 0])
        else:
            for i in range(time_steps, test_scaled.shape[0]):
                x_test.append(test_scaled[i - time_steps:i])
                y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    def prepare_return_multitask_data(self, time_steps, project_folder, validation_fraction=0.15):
        """Prepare v9 scaled log-return windows with return and direction targets."""
        (x_train, y_return_train), (x_test, y_return_test), (training_data, test_data) = self.download_transform_to_numpy(
            time_steps,
            project_folder,
            use_returns=True,
            use_deltas=False,
            use_trend_residual=False,
            forecast_horizon=1,
            validation_fraction=validation_fraction,
        )

        train_returns = self._min_max.inverse_transform(
            np.asarray(y_return_train).reshape(-1, 1)
        ).reshape(-1)
        test_returns = self._min_max.inverse_transform(
            np.asarray(y_return_test).reshape(-1, 1)
        ).reshape(-1)

        y_direction_train = (train_returns > 0.0).astype(np.float32)
        y_direction_test = (test_returns > 0.0).astype(np.float32)

        return (
            x_train,
            y_direction_train,
            np.asarray(y_return_train, dtype=np.float32),
        ), (
            x_test,
            y_direction_test,
            np.asarray(y_return_test, dtype=np.float32),
        ), (
            training_data,
            test_data,
        )

    def prepare_delta_direction_data(self, time_steps, validation_date, validation_fraction=0.15):
        end_date = datetime.today()
        data = self._download_close_frame(end_date).reset_index()
        data = data.set_index('Date')

        training_data = data[data.index < validation_date].copy()
        test_data = data[data.index >= validation_date].copy()

        close_series = pd.concat([training_data['Close'], test_data['Close']])
        deltas = self._compute_deltas(close_series)
        training_deltas = deltas[deltas.index < validation_date]
        test_deltas = deltas[deltas.index >= validation_date]

        fit_count = self._fit_sample_count(len(training_deltas) - time_steps, validation_fraction)
        fit_end = time_steps + fit_count
        self._input_scaler.fit(training_data.iloc[:fit_end])
        self._min_max.fit(training_deltas.abs().iloc[:fit_end].to_frame())
        close_scaled_all = self._input_scaler.transform(pd.concat((training_data, test_data), axis=0))
        close_scaled_aligned = close_scaled_all[1:]

        magnitude = training_deltas.abs()
        mag_scaled = self._min_max.transform(self._ensure_frame(magnitude))

        total_magnitude = pd.concat((training_deltas.abs(), test_deltas.abs()), axis=0)
        total_mag_scaled = self._min_max.transform(self._ensure_frame(total_magnitude))

        direction = (training_deltas > 0).astype(np.float32)
        total_direction = (pd.concat((training_deltas, test_deltas), axis=0) > 0).astype(np.float32)

        x_train = []
        y_dir_train = []
        y_mag_train = []
        for i in range(time_steps, len(training_deltas)):
            x_train.append(close_scaled_aligned[i - time_steps:i])
            y_dir_train.append(direction.iloc[i])
            y_mag_train.append(mag_scaled[i, 0])

        x_test = []
        y_dir_test = []
        y_mag_test = []
        test_start = len(training_deltas)
        for i in range(test_start, len(total_magnitude)):
            x_test.append(close_scaled_aligned[i - time_steps:i])
            y_dir_test.append(total_direction.iloc[i])
            y_mag_test.append(total_mag_scaled[i, 0])

        x_train = np.array(x_train)
        y_dir_train = np.array(y_dir_train)
        y_mag_train = np.array(y_mag_train)
        x_test = np.array(x_test)
        y_dir_test = np.array(y_dir_test)
        y_mag_test = np.array(y_mag_test)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return (x_train, y_dir_train, y_mag_train), (x_test, y_dir_test, y_mag_test), (training_data, test_data)

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        original_price = latest_close_price

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            original_price = original_price + (original_price * random_slope)
            #print(original_price)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)

        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        #print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            #print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data



